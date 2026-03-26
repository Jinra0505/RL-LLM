from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from mock_recovery_env import MockRecoveryEnv

LOGGER = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_revise_functions(module_path: Path) -> tuple[Callable[..., np.ndarray], Callable[..., float]]:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import revise module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    revise_state = getattr(module, "revise_state", None)
    intrinsic_reward = getattr(module, "intrinsic_reward", None)
    if not callable(revise_state) or not callable(intrinsic_reward):
        raise ValueError(f"Module {module_path} must define revise_state and intrinsic_reward")
    return revise_state, intrinsic_reward


def _call_revise(fn: Callable[..., Any], state: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    try:
        return np.asarray(fn(state, info), dtype=float)
    except TypeError:
        return np.asarray(fn(state), dtype=float)


def _call_intrinsic(
    fn: Callable[..., Any],
    state: np.ndarray,
    action: int,
    next_state: np.ndarray,
    info: dict[str, Any],
    revised_state: np.ndarray,
) -> float:
    try:
        return float(fn(state, action, next_state, info, revised_state))
    except TypeError:
        try:
            return float(fn(revised_state))
        except TypeError:
            return float(fn(next_state))


def run_training(
    revise_module_path: Path | None,
    env_name: str,
    train_episodes: int,
    eval_episodes: int,
    max_steps_per_episode: int,
    gamma: float,
    task_mode: str,
    llm_mode: str,
    output_json_path: Path,
    seed: int = 42,
) -> dict[str, Any]:
    if env_name != "mock_recovery":
        raise ValueError("This minimal prototype supports only env_name=mock_recovery")

    revise_module_path = revise_module_path or Path("baseline_noop.py")
    revise_state, intrinsic_reward = load_revise_functions(revise_module_path)
    env = MockRecoveryEnv(max_steps=max_steps_per_episode, seed=seed)

    q_table: dict[tuple[int, ...], np.ndarray] = {}
    epsilon, alpha = 0.3, 0.2
    action_usage: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

    def key_fn(x: np.ndarray) -> tuple[int, ...]:
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        return tuple((x * 5).astype(int).tolist())

    episode_rewards: list[float] = []
    eval_rewards: list[float] = []
    successes = 0
    comm_scores: list[float] = []
    crit_scores: list[float] = []
    trans_scores: list[float] = []
    resource_values: list[float] = []
    completion_steps: list[int] = []
    violation_count = 0
    progress_deltas: list[float] = []
    stage_acc = {"early_recovery": 0.0, "mid_recovery": 0.0, "late_recovery": 0.0}
    stage_cnt = {"early_recovery": 0, "mid_recovery": 0, "late_recovery": 0}

    np.random.seed(seed)
    for ep in range(train_episodes):
        state, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        for step in range(max_steps_per_episode):
            rs = _call_revise(revise_state, state, info)
            k = key_fn(rs[:7])
            if k not in q_table:
                q_table[k] = np.zeros(4, dtype=float)
            if np.random.rand() < epsilon:
                action = int(np.random.randint(0, 4))
            else:
                action = int(np.argmax(q_table[k]))
            action_usage[action] += 1

            next_state, env_reward, terminated, truncated, info = env.step(action)
            shaped = _call_intrinsic(intrinsic_reward, state, action, next_state, info, rs)
            reward = float(env_reward + shaped)
            next_rs = _call_revise(revise_state, next_state, info)
            nk = key_fn(next_rs[:7])
            if nk not in q_table:
                q_table[nk] = np.zeros(4, dtype=float)

            done = bool(terminated or truncated)
            target = reward + gamma * (0.0 if done else np.max(q_table[nk]))
            q_table[k][action] += alpha * (target - q_table[k][action])

            ep_reward += reward
            state = next_state
            if info.get("constraint_violation", False):
                violation_count += 1
            progress_deltas.append(float(info.get("progress_delta", 0.0)))
            stage = str(info.get("stage", "early_recovery"))
            if stage in stage_acc:
                stage_acc[stage] += float(info.get("progress_delta", 0.0))
                stage_cnt[stage] += 1

            if done:
                if terminated:
                    successes += 1
                    completion_steps.append(step + 1)
                break
        episode_rewards.append(ep_reward)
        epsilon = max(0.05, epsilon * 0.98)

    for ep in range(eval_episodes):
        state, info = env.reset(seed=seed + 1000 + ep)
        total = 0.0
        for _ in range(max_steps_per_episode):
            rs = _call_revise(revise_state, state, info)
            k = key_fn(rs[:7])
            if k not in q_table:
                q_table[k] = np.zeros(4, dtype=float)
            action = int(np.argmax(q_table[k]))
            next_state, env_reward, terminated, truncated, info = env.step(action)
            total += float(env_reward)
            state = next_state
            if terminated or truncated:
                break
        eval_rewards.append(total)
        comm_scores.append(float(info.get("communication_recovery_level", 0.0)))
        crit_scores.append(float(info.get("critical_load_recovery_level", 0.0)))
        trans_scores.append(float(info.get("transportation_accessibility", 0.0)))
        resource_values.append(float(info.get("available_repair_resources", 0.0)))

    total_actions = max(1, sum(action_usage.values()))
    result = {
        "episode_rewards": episode_rewards,
        "eval_rewards": eval_rewards,
        "success_rate": successes / max(1, train_episodes),
        "communication_recovery_ratio": float(np.mean(comm_scores)) if comm_scores else 0.0,
        "critical_load_recovery_ratio": float(np.mean(crit_scores)) if crit_scores else 0.0,
        "transportation_recovery_ratio": float(np.mean(trans_scores)) if trans_scores else 0.0,
        "recovery_completion_time": float(np.mean(completion_steps)) if completion_steps else float(max_steps_per_episode),
        "resource_utilization": float(1.0 - np.mean(resource_values)) if resource_values else 0.0,
        "constraint_violation_count": int(violation_count),
        "mean_progress_delta": float(np.mean(progress_deltas)) if progress_deltas else 0.0,
        "stage_progress": {k: (stage_acc[k] / stage_cnt[k] if stage_cnt[k] else 0.0) for k in stage_acc},
        "task_mode_used": task_mode,
        "llm_mode_used": llm_mode,
        "revise_module_path": str(revise_module_path),
        "env_name": env_name,
        "action_usage": {str(k): v / total_actions for k, v in action_usage.items()},
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("Saved RL results to %s", output_json_path)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Minimal RL trainer for generated revise/reward modules.")
    parser.add_argument("--env", default="mock_recovery")
    parser.add_argument("--revise-module", default="")
    parser.add_argument("--task-mode", default="system_recovery_priority")
    parser.add_argument("--llm-mode", default="auto")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="outputs/rl_result.json")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    tr = cfg["training"]
    revise_module_path = Path(args.revise_module) if args.revise_module else None
    run_training(
        revise_module_path=revise_module_path,
        env_name=args.env,
        train_episodes=int(tr["train_episodes"]),
        eval_episodes=int(tr["eval_episodes"]),
        max_steps_per_episode=int(tr["max_steps_per_episode"]),
        gamma=float(tr["gamma"]),
        task_mode=args.task_mode,
        llm_mode=args.llm_mode,
        output_json_path=Path(args.output),
    )


if __name__ == "__main__":
    main()

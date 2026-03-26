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
        arr = np.asarray(fn(state, info), dtype=float)
    except TypeError:
        arr = np.asarray(fn(state), dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


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


def _effective_state(revised_state: np.ndarray, max_revised_dim: int | None) -> np.ndarray:
    """Policy features are based on revised_state, not raw state.

    If max_revised_dim is None, all revised dimensions are used (default).
    """
    if max_revised_dim is None:
        return revised_state
    return revised_state[:max_revised_dim]


def _weighted_mode_score(metrics: dict[str, Any], task_mode: str, weights_cfg: dict[str, Any]) -> float:
    weights = dict(weights_cfg.get(task_mode, {}))
    if not weights:
        return float(metrics.get("success_rate", 0.0))
    score = 0.0
    for metric, w in weights.items():
        score += float(w) * float(metrics.get(metric, 0.0))
    return float(score)


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
    max_revised_dim: int | None = None,
    task_mode_metric_weights: dict[str, Any] | None = None,
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
        x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
        return tuple((x * 5).astype(int).tolist())

    episode_rewards: list[float] = []
    eval_rewards: list[float] = []
    successes = 0
    comm_scores: list[float] = []
    power_scores: list[float] = []
    trans_scores: list[float] = []
    resource_values: list[float] = []
    completion_steps: list[int] = []
    violation_count = 0
    progress_deltas: list[float] = []
    stage_acc = {"early": 0.0, "middle": 0.0, "late": 0.0}
    stage_cnt = {"early": 0, "middle": 0, "late": 0}

    np.random.seed(seed)
    for ep in range(train_episodes):
        state, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        for step in range(max_steps_per_episode):
            rs = _call_revise(revise_state, state, info)
            policy_state = _effective_state(rs, max_revised_dim)
            k = key_fn(policy_state)
            if k not in q_table:
                q_table[k] = np.zeros(4, dtype=float)

            action = int(np.random.randint(0, 4)) if np.random.rand() < epsilon else int(np.argmax(q_table[k]))
            action_usage[action] += 1

            next_state, env_reward, terminated, truncated, info = env.step(action)
            shaped = _call_intrinsic(intrinsic_reward, state, action, next_state, info, rs)
            reward = float(env_reward + shaped)

            next_rs = _call_revise(revise_state, next_state, info)
            next_policy_state = _effective_state(next_rs, max_revised_dim)
            nk = key_fn(next_policy_state)
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
            stage = str(info.get("stage", "early"))
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
            policy_state = _effective_state(rs, max_revised_dim)
            k = key_fn(policy_state)
            if k not in q_table:
                q_table[k] = np.zeros(4, dtype=float)
            action = int(np.argmax(q_table[k]))
            next_state, env_reward, terminated, truncated, info = env.step(action)
            total += float(env_reward)
            state = next_state
            if terminated or truncated:
                break
        eval_rewards.append(total)
        comm_scores.append(float(info.get("communication_recovery_ratio", 0.0)))
        power_scores.append(float(info.get("power_recovery_ratio", 0.0)))
        trans_scores.append(float(info.get("transportation_accessibility_ratio", 0.0)))
        resource_values.append(float(info.get("available_repair_resources", 0.0)))

    total_actions = max(1, sum(action_usage.values()))
    result = {
        "episode_rewards": episode_rewards,
        "eval_rewards": eval_rewards,
        "success_rate": successes / max(1, train_episodes),
        "communication_recovery_ratio": float(np.mean(comm_scores)) if comm_scores else 0.0,
        "power_recovery_ratio": float(np.mean(power_scores)) if power_scores else 0.0,
        "transportation_recovery_ratio": float(np.mean(trans_scores)) if trans_scores else 0.0,
        "recovery_completion_time": float(np.mean(completion_steps)) if completion_steps else float(max_steps_per_episode),
        "resource_utilization": float(1.0 - np.mean(resource_values)) if resource_values else 0.0,
        "constraint_violation_count": int(violation_count),
        "mean_progress_delta": float(np.mean(progress_deltas)) if progress_deltas else 0.0,
        "stage_progress": {k: (stage_acc[k] / stage_cnt[k] if stage_cnt[k] else 0.0) for k in stage_acc},
        "cumulative_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "task_mode_used": task_mode,
        "llm_mode_used": llm_mode,
        "revise_module_path": str(revise_module_path),
        "env_name": env_name,
        "policy_feature_dim_used": int(_effective_state(_call_revise(revise_state, np.zeros(7), {}), max_revised_dim).shape[0]),
        "action_usage": {str(k): v / total_actions for k, v in action_usage.items()},
    }

    weights_cfg = task_mode_metric_weights or {}
    result["selection_score"] = _weighted_mode_score(result, task_mode=task_mode, weights_cfg=weights_cfg)
    result["selection_metric_used"] = f"weighted_score::{task_mode}"

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
    max_dim = cfg.get("state_representation", {}).get("max_revised_dim", None)
    weights = cfg.get("selection", {}).get("task_mode_metric_weights", {})
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
        max_revised_dim=int(max_dim) if max_dim is not None else None,
        task_mode_metric_weights=weights,
    )


if __name__ == "__main__":
    main()

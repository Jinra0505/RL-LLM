from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from envs.mock_recovery_env import MockRecoveryEnv

LOGGER = logging.getLogger(__name__)


@dataclass
class ReviseFunctions:
    revise_state: Callable[[np.ndarray, dict[str, Any] | None], np.ndarray]
    intrinsic_reward: Callable[[np.ndarray, int, np.ndarray, dict[str, Any] | None, np.ndarray | None], float]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_revise_module(module_path: Path) -> ReviseFunctions:
    module_path = module_path.resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import revise module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    revise_state = getattr(module, "revise_state", None)
    intrinsic_reward = getattr(module, "intrinsic_reward", None)
    if not callable(revise_state) or not callable(intrinsic_reward):
        raise ValueError(f"Module {module_path} must define callable revise_state and intrinsic_reward")

    def revise_wrapper(state: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        try:
            return np.asarray(revise_state(state, info), dtype=float)
        except TypeError:
            return np.asarray(revise_state(state), dtype=float)

    def reward_wrapper(
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: dict[str, Any] | None = None,
        revised_state: np.ndarray | None = None,
    ) -> float:
        try:
            return float(intrinsic_reward(state, action, next_state, info, revised_state))
        except TypeError:
            try:
                rs = revised_state if revised_state is not None else revise_wrapper(next_state, info)
                return float(intrinsic_reward(rs))
            except TypeError:
                return float(intrinsic_reward(next_state))

    return ReviseFunctions(revise_wrapper, reward_wrapper)


def epsilon_greedy_action(q_table: dict[tuple[float, ...], np.ndarray], key: tuple[float, ...], epsilon: float) -> int:
    if key not in q_table:
        q_table[key] = np.zeros(4, dtype=float)
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, 4))
    return int(np.argmax(q_table[key]))


def bucketize(x: np.ndarray, bins: int = 5) -> tuple[float, ...]:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return tuple((x * bins).astype(int).tolist())


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
    np.random.seed(seed)
    if env_name != "mock_recovery":
        raise ValueError(f"Unsupported env: {env_name}")

    if revise_module_path is None:
        revise_module_path = Path("baseline_noop.py")

    hooks = load_revise_module(revise_module_path)
    env = MockRecoveryEnv(max_steps=max_steps_per_episode, seed=seed)

    q_table: dict[tuple[float, ...], np.ndarray] = {}
    alpha = 0.2
    epsilon = 0.3

    episode_rewards: list[float] = []
    eval_rewards: list[float] = []
    successes = 0
    comm_scores: list[float] = []
    crit_scores: list[float] = []
    completion_steps: list[int] = []
    resource_values: list[float] = []
    violation_count = 0

    for ep in range(train_episodes):
        state, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        for step in range(max_steps_per_episode):
            rs = hooks.revise_state(state, info)
            key = bucketize(rs[:7])
            action = epsilon_greedy_action(q_table, key, epsilon)
            next_state, env_reward, terminated, truncated, info = env.step(action)
            shaped = hooks.intrinsic_reward(state, action, next_state, info, rs)
            reward = float(env_reward + shaped)
            next_key = bucketize(hooks.revise_state(next_state, info)[:7])
            if next_key not in q_table:
                q_table[next_key] = np.zeros(4, dtype=float)
            td_target = reward + gamma * (0.0 if (terminated or truncated) else np.max(q_table[next_key]))
            q_table[key][action] += alpha * (td_target - q_table[key][action])
            ep_reward += reward
            state = next_state
            if info.get("constraint_violation", False):
                violation_count += 1
            if terminated or truncated:
                if terminated:
                    successes += 1
                    completion_steps.append(step + 1)
                break
        episode_rewards.append(ep_reward)
        epsilon = max(0.05, epsilon * 0.98)

    for ep in range(eval_episodes):
        state, info = env.reset(seed=seed + 1000 + ep)
        total = 0.0
        for step in range(max_steps_per_episode):
            rs = hooks.revise_state(state, info)
            key = bucketize(rs[:7])
            if key not in q_table:
                q_table[key] = np.zeros(4, dtype=float)
            action = int(np.argmax(q_table[key]))
            next_state, env_reward, terminated, truncated, info = env.step(action)
            total += float(env_reward)
            state = next_state
            if terminated or truncated:
                break
        eval_rewards.append(total)
        comm_scores.append(float(info["communication_recovery_level"]))
        crit_scores.append(float(info["critical_load_recovery_level"]))
        resource_values.append(float(info["available_repair_resources"]))

    result = {
        "episode_rewards": episode_rewards,
        "eval_rewards": eval_rewards,
        "success_rate": successes / float(max(1, train_episodes)),
        "communication_recovery_ratio": float(np.mean(comm_scores)) if comm_scores else 0.0,
        "critical_load_recovery_ratio": float(np.mean(crit_scores)) if crit_scores else 0.0,
        "recovery_completion_time": float(np.mean(completion_steps)) if completion_steps else float(max_steps_per_episode),
        "resource_utilization": float(1.0 - np.mean(resource_values)) if resource_values else 0.0,
        "constraint_violation_count": int(violation_count),
        "task_mode_used": task_mode,
        "llm_mode_used": llm_mode,
        "revise_module_path": str(revise_module_path),
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("Saved RL results to %s", output_json_path)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train RL inner loop with generated revise/reward module.")
    parser.add_argument("--env", default="mock_recovery")
    parser.add_argument("--revise-module", default="")
    parser.add_argument("--task-mode", default="system_recovery_priority")
    parser.add_argument("--llm-mode", default="auto")
    parser.add_argument("--config", default="configs/env_schema.yaml")
    parser.add_argument("--output", default="outputs/rl_result.json")
    args = parser.parse_args()

    env_cfg = load_yaml(Path(args.config)).get("training_defaults", {})
    revise_module_path = Path(args.revise_module) if args.revise_module else None
    run_training(
        revise_module_path=revise_module_path,
        env_name=args.env,
        train_episodes=int(env_cfg.get("train_episodes", 20)),
        eval_episodes=int(env_cfg.get("eval_episodes", 8)),
        max_steps_per_episode=int(env_cfg.get("max_steps_per_episode", 40)),
        gamma=float(env_cfg.get("gamma", 0.95)),
        task_mode=args.task_mode,
        llm_mode=args.llm_mode,
        output_json_path=Path(args.output),
    )


if __name__ == "__main__":
    main()

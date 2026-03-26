from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from mock_recovery_env import ProjectRecoveryEnv

LOGGER = logging.getLogger(__name__)


class QNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        self.buf.append((s, a, r, ns, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.stack(s), np.array(a), np.array(r, dtype=np.float32), np.stack(ns), np.array(d, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.buf)


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
        arr = np.asarray(fn(state, info), dtype=np.float32)
    except TypeError:
        arr = np.asarray(fn(state), dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def _call_intrinsic(fn: Callable[..., Any], s: np.ndarray, a: int, ns: np.ndarray, info: dict[str, Any], rs: np.ndarray) -> float:
    try:
        return float(fn(s, a, ns, info, rs))
    except TypeError:
        try:
            return float(fn(rs))
        except TypeError:
            return float(fn(ns))


def _effective_state(revised_state: np.ndarray, max_revised_dim: int | None) -> np.ndarray:
    """Policy input is revised_state (LLM can affect representation learning)."""
    return revised_state if max_revised_dim is None else revised_state[:max_revised_dim]


def _weighted_mode_score(metrics: dict[str, Any], task_mode: str, weights_cfg: dict[str, Any]) -> float:
    weights = dict(weights_cfg.get(task_mode, {}))
    if not weights:
        return float(metrics.get("success_rate", 0.0))
    return float(sum(float(w) * float(metrics.get(k, 0.0)) for k, w in weights.items()))


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
    dqn_cfg: dict[str, Any] | None = None,
    severity: str = "moderate",
) -> dict[str, Any]:
    if env_name not in {"project_recovery", "mock_recovery"}:
        raise ValueError("Supported env names: project_recovery or mock_recovery")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    revise_module_path = revise_module_path or Path("baseline_noop.py")
    revise_state_fn, intrinsic_reward_fn = load_revise_functions(revise_module_path)

    env = ProjectRecoveryEnv(max_steps=max_steps_per_episode, seed=seed, severity=severity)
    action_dim = int(env.action_space.n)

    dqn_cfg = dqn_cfg or {}
    batch_size = int(dqn_cfg.get("batch_size", 64))
    replay_size = int(dqn_cfg.get("replay_size", 30000))
    min_replay_size = int(dqn_cfg.get("min_replay_size", 500))
    train_freq = int(dqn_cfg.get("train_freq", 1))
    target_update_interval = int(dqn_cfg.get("target_update_interval", 250))
    lr = float(dqn_cfg.get("learning_rate", 8e-4))
    hidden_dim = int(dqn_cfg.get("hidden_dim", 128))
    eps_start = float(dqn_cfg.get("epsilon_start", 1.0))
    eps_end = float(dqn_cfg.get("epsilon_end", 0.05))
    eps_decay_steps = int(dqn_cfg.get("epsilon_decay_steps", 5000))

    s0, info0 = env.reset(seed=seed)
    rs0 = _effective_state(_call_revise(revise_state_fn, s0, info0), max_revised_dim)
    state_dim = int(rs0.shape[0])

    q_net = QNet(state_dim, action_dim, hidden_dim)
    target_net = QNet(state_dim, action_dim, hidden_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(replay_size)

    global_step = 0
    episode_rewards: list[float] = []
    eval_rewards: list[float] = []
    action_usage = {str(i): 0 for i in range(action_dim)}

    successes = 0
    completion_steps: list[int] = []
    comm_scores: list[float] = []
    power_scores: list[float] = []
    road_scores: list[float] = []
    crit_scores: list[float] = []
    violations = 0

    for ep in range(train_episodes):
        s, info = env.reset(seed=seed + ep)
        ep_reward = 0.0

        for step in range(max_steps_per_episode):
            rs = _effective_state(_call_revise(revise_state_fn, s, info), max_revised_dim)

            eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - global_step / float(max(1, eps_decay_steps)))
            if random.random() < eps:
                a = int(np.random.randint(0, action_dim))
            else:
                with torch.no_grad():
                    qvals = q_net(torch.tensor(rs, dtype=torch.float32).unsqueeze(0))
                    a = int(torch.argmax(qvals, dim=1).item())

            action_usage[str(a)] += 1
            ns, ext_r, terminated, truncated, info = env.step(a)
            ir = _call_intrinsic(intrinsic_reward_fn, s, a, ns, info, rs)
            r = float(ext_r + ir)
            done = bool(terminated or truncated)

            nrs = _effective_state(_call_revise(revise_state_fn, ns, info), max_revised_dim)
            replay.add(rs, a, r, nrs, done)

            if len(replay) >= min_replay_size and global_step % train_freq == 0:
                bs, ba, br, bns, bd = replay.sample(batch_size)
                bs_t = torch.tensor(bs, dtype=torch.float32)
                ba_t = torch.tensor(ba, dtype=torch.int64).unsqueeze(1)
                br_t = torch.tensor(br, dtype=torch.float32).unsqueeze(1)
                bns_t = torch.tensor(bns, dtype=torch.float32)
                bd_t = torch.tensor(bd, dtype=torch.float32).unsqueeze(1)

                q = q_net(bs_t).gather(1, ba_t)
                with torch.no_grad():
                    max_next_q = target_net(bns_t).max(dim=1, keepdim=True).values
                    target = br_t + gamma * (1.0 - bd_t) * max_next_q
                loss = nn.functional.smooth_l1_loss(q, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
                optimizer.step()

            if global_step % target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

            ep_reward += r
            s = ns
            global_step += 1

            if info.get("constraint_violation", False):
                violations += 1
            if done:
                if terminated:
                    successes += 1
                    completion_steps.append(step + 1)
                break

        episode_rewards.append(ep_reward)

    # Evaluation (greedy)
    for ep in range(eval_episodes):
        s, info = env.reset(seed=seed + 1000 + ep)
        total = 0.0
        for _ in range(max_steps_per_episode):
            rs = _effective_state(_call_revise(revise_state_fn, s, info), max_revised_dim)
            with torch.no_grad():
                qvals = q_net(torch.tensor(rs, dtype=torch.float32).unsqueeze(0))
                a = int(torch.argmax(qvals, dim=1).item())
            ns, ext_r, terminated, truncated, info = env.step(a)
            total += float(ext_r)
            s = ns
            if terminated or truncated:
                break

        eval_rewards.append(total)
        comm_scores.append(float(info.get("communication_recovery_ratio", 0.0)))
        power_scores.append(float(info.get("power_recovery_ratio", 0.0)))
        road_scores.append(float(info.get("road_recovery_ratio", 0.0)))
        crit_scores.append(float(info.get("critical_load_recovery_ratio", 0.0)))

    total_actions = max(1, sum(action_usage.values()))
    result = {
        "episode_rewards": episode_rewards,
        "eval_rewards": eval_rewards,
        "success_rate": successes / float(max(1, train_episodes)),
        "communication_recovery_ratio": float(np.mean(comm_scores)) if comm_scores else 0.0,
        "power_recovery_ratio": float(np.mean(power_scores)) if power_scores else 0.0,
        "road_recovery_ratio": float(np.mean(road_scores)) if road_scores else 0.0,
        "critical_load_recovery_ratio": float(np.mean(crit_scores)) if crit_scores else 0.0,
        "recovery_completion_time": float(np.mean(completion_steps)) if completion_steps else float(max_steps_per_episode),
        "cumulative_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "constraint_violation_count": int(violations),
        "task_mode_used": task_mode,
        "llm_mode_used": llm_mode,
        "revise_module_path": str(revise_module_path),
        "policy_feature_dim_used": state_dim,
        "env_name": env_name,
        "severity": severity,
        "action_usage": {k: v / total_actions for k, v in action_usage.items()},
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
    parser = argparse.ArgumentParser(description="Lightweight DQN trainer for project recovery env.")
    parser.add_argument("--env", default="project_recovery")
    parser.add_argument("--revise-module", default="")
    parser.add_argument("--task-mode", default="coordinated_restoration")
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
        max_steps_per_episode=int(cfg["env"].get("max_steps", 60)),
        gamma=float(tr["gamma"]),
        task_mode=args.task_mode,
        llm_mode=args.llm_mode,
        output_json_path=Path(args.output),
        max_revised_dim=int(max_dim) if max_dim is not None else None,
        task_mode_metric_weights=weights,
        dqn_cfg=tr,
        severity=str(cfg.get("scenario", {}).get("severity", "moderate")),
    )


if __name__ == "__main__":
    main()

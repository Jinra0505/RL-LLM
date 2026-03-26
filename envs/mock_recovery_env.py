from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MockRecoveryEnv(gym.Env):
    """Minimal Gymnasium env for coordinated recovery experiments."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 40, seed: int | None = None) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        self.state = np.zeros(7, dtype=np.float32)
        self.step_count = 0
        self.constraint_violation_count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        _ = options
        self.step_count = 0
        self.constraint_violation_count = 0
        comm = self.rng.uniform(0.15, 0.4)
        crit = self.rng.uniform(0.1, 0.35)
        trans = self.rng.uniform(0.2, 0.5)
        resources = self.rng.uniform(0.5, 0.9)
        storage = self.rng.uniform(0.4, 0.8)
        stage = 0.0
        flag = 0.0
        self.state = np.array([comm, crit, trans, resources, storage, stage, flag], dtype=np.float32)
        return self.state.copy(), self._build_info()

    def step(self, action: int):
        action = int(action)
        comm, crit, trans, resources, storage, _stage, _flag = self.state.tolist()
        noise = self.rng.normal(0.0, 0.01, size=3)

        if action == 0:
            gains = np.array([0.06, 0.02, 0.015])
            resources -= 0.05
            storage -= 0.02
        elif action == 1:
            gains = np.array([0.02, 0.065, 0.015])
            resources -= 0.06
            storage -= 0.01
        elif action == 2:
            gains = np.array([0.015, 0.02, 0.06])
            resources -= 0.05
            storage -= 0.03
        else:
            gains = np.array([0.035, 0.04, 0.035])
            resources -= 0.04
            storage -= 0.02

        comm = float(np.clip(comm + gains[0] + noise[0], 0.0, 1.0))
        crit = float(np.clip(crit + gains[1] + noise[1], 0.0, 1.0))
        trans = float(np.clip(trans + gains[2] + noise[2], 0.0, 1.0))
        resources = float(np.clip(resources + 0.02, 0.0, 1.0))
        storage = float(np.clip(storage + 0.01, 0.0, 1.0))

        stage = 0.0 if (comm + crit + trans) / 3 < 0.35 else 1.0 if (comm + crit + trans) / 3 < 0.75 else 2.0
        flag = 1.0 if (resources < 0.12 or storage < 0.1) else 0.0
        if flag > 0.5:
            self.constraint_violation_count += 1

        self.state = np.array([comm, crit, trans, resources, storage, stage / 2.0, flag], dtype=np.float32)
        self.step_count += 1

        base_reward = 0.4 * comm + 0.45 * crit + 0.15 * trans - 0.1 * flag
        terminated = bool(comm >= 0.95 and crit >= 0.95 and trans >= 0.90)
        truncated = self.step_count >= self.max_steps

        return self.state.copy(), float(base_reward), terminated, truncated, self._build_info()

    def _build_info(self) -> dict[str, Any]:
        comm, crit, trans, resources, _storage, stage, flag = self.state.tolist()
        return {
            "communication_recovery_level": float(comm),
            "critical_load_recovery_level": float(crit),
            "transportation_accessibility": float(trans),
            "available_repair_resources": float(resources),
            "stage": "early_recovery" if stage < 0.3 else "mid_recovery" if stage < 0.8 else "late_recovery",
            "constraint_violation": bool(flag > 0.5),
            "constraint_violation_count": self.constraint_violation_count,
        }

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MockRecoveryEnv(gym.Env):
    """Minimal stage-aware environment for end-to-end prototype runs."""

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

    def _stage_from_progress(self, progress: float) -> tuple[float, str]:
        if progress < 0.35:
            return 0.0, "early_recovery"
        if progress < 0.75:
            return 0.5, "mid_recovery"
        return 1.0, "late_recovery"

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _ = options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.constraint_violation_count = 0
        comm = self.rng.uniform(0.12, 0.32)
        crit = self.rng.uniform(0.08, 0.28)
        trans = self.rng.uniform(0.15, 0.40)
        resources = self.rng.uniform(0.50, 0.85)
        storage = self.rng.uniform(0.35, 0.75)
        self.state = np.array([comm, crit, trans, resources, storage, 0.0, 0.0], dtype=np.float32)
        return self.state.copy(), self._build_info(progress_delta=0.0, invalid_action=False, resource_shortage=False)

    def step(self, action: int):
        action = int(action)
        comm, crit, trans, resources, storage, _stage, _flag = self.state.tolist()
        old_progress = (comm + crit + trans) / 3.0
        _, stage_name = self._stage_from_progress(old_progress)

        invalid_action = False
        resource_shortage = resources < 0.15 or storage < 0.12

        if action == 0:
            gains = np.array([0.055, 0.015, 0.005])
            resources -= 0.06
            storage -= 0.025
        elif action == 1:
            gains = np.array([0.01, 0.07, 0.005])
            resources -= 0.07
            storage -= 0.015
        elif action == 2:
            gains = np.array([0.01, 0.01, 0.065])
            resources -= 0.055
            storage -= 0.03
        else:
            gains = np.array([0.03, 0.035, 0.03])
            resources -= 0.045
            storage -= 0.02

        if action == 2 and stage_name == "early_recovery":
            gains *= 0.25
            invalid_action = True
        if resource_shortage and action in (0, 1, 2):
            gains *= 0.2
            invalid_action = True

        noise = self.rng.normal(0.0, 0.008, size=3)
        comm = float(np.clip(comm + gains[0] + noise[0], 0.0, 1.0))
        crit = float(np.clip(crit + gains[1] + noise[1], 0.0, 1.0))
        trans = float(np.clip(trans + gains[2] + noise[2], 0.0, 1.0))

        replenish = 0.012 if not resource_shortage else 0.004
        resources = float(np.clip(resources + replenish, 0.0, 1.0))
        storage = float(np.clip(storage + replenish * 0.8, 0.0, 1.0))

        progress = (comm + crit + trans) / 3.0
        stage_value, _ = self._stage_from_progress(progress)
        constraint_flag = 1.0 if (resource_shortage or invalid_action) else 0.0
        if constraint_flag > 0.5:
            self.constraint_violation_count += 1

        self.state = np.array([comm, crit, trans, resources, storage, stage_value, constraint_flag], dtype=np.float32)
        self.step_count += 1

        progress_delta = float(progress - old_progress)
        reward = 0.35 * comm + 0.45 * crit + 0.2 * trans + 0.8 * progress_delta
        if invalid_action:
            reward -= 0.08
        if resource_shortage:
            reward -= 0.06

        terminated = bool(comm >= 0.95 and crit >= 0.95 and trans >= 0.92)
        truncated = self.step_count >= self.max_steps
        info = self._build_info(progress_delta=progress_delta, invalid_action=invalid_action, resource_shortage=resource_shortage)
        return self.state.copy(), float(reward), terminated, truncated, info

    def _build_info(self, progress_delta: float, invalid_action: bool, resource_shortage: bool) -> dict[str, Any]:
        comm, crit, trans, resources, _storage, stage_value, flag = self.state.tolist()
        stage = "early_recovery" if stage_value < 0.3 else "mid_recovery" if stage_value < 0.8 else "late_recovery"
        return {
            "progress_delta": float(progress_delta),
            "invalid_action": bool(invalid_action),
            "resource_shortage": bool(resource_shortage),
            "stage": stage,
            "constraint_violation": bool(flag > 0.5),
            "constraint_violation_count": int(self.constraint_violation_count),
            "communication_recovery_level": float(comm),
            "critical_load_recovery_level": float(crit),
            "transportation_accessibility": float(trans),
            "available_repair_resources": float(resources),
        }

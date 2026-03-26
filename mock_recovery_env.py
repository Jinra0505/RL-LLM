from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MockRecoveryEnv(gym.Env):
    """Reduced-order three-layer coupled recovery environment.

    State (7-dim):
      0 communication_recovery_ratio
      1 power_recovery_ratio (critical-load restoration proxy)
      2 transportation_accessibility_ratio
      3 available_repair_resources
      4 mobile_energy_storage_level
      5 stage_indicator (0 early, 0.5 middle, 1 late)
      6 constraint_flag
    """

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
            return 0.0, "early"
        if progress < 0.75:
            return 0.5, "middle"
        return 1.0, "late"

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _ = options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.constraint_violation_count = 0
        comm = self.rng.uniform(0.12, 0.30)
        power = self.rng.uniform(0.10, 0.28)
        transport = self.rng.uniform(0.15, 0.40)
        resources = self.rng.uniform(0.50, 0.85)
        storage = self.rng.uniform(0.35, 0.75)
        self.state = np.array([comm, power, transport, resources, storage, 0.0, 0.0], dtype=np.float32)
        return self.state.copy(), self._build_info(progress_delta=0.0, invalid_action=False, resource_shortage=False)

    def step(self, action: int):
        action = int(action)
        comm, power, transport, resources, storage, _stage, _flag = self.state.tolist()
        old_progress = (comm + power + transport) / 3.0
        _, stage_name = self._stage_from_progress(old_progress)

        resource_shortage = resources < 0.15 or storage < 0.12
        invalid_action = False

        # Base gains: [communication, power, transport]
        if action == 0:  # prioritize communication
            gains = np.array([0.060, 0.010, 0.005])
            res_cost, stor_cost = 0.060, 0.020
        elif action == 1:  # prioritize power
            gains = np.array([0.010, 0.065, 0.005])
            res_cost, stor_cost = 0.070, 0.025
        elif action == 2:  # prioritize transportation
            gains = np.array([0.005, 0.005, 0.070])
            res_cost, stor_cost = 0.055, 0.018
        else:  # balanced/coordinated
            gains = np.array([0.030, 0.030, 0.030])
            res_cost, stor_cost = 0.050, 0.020

        # Coupling 1: Power supports communication.
        if action == 0 and power < 0.35:
            gains[0] *= 0.55

        # Coupling 2: Transportation supports communication and power restoration.
        if action in (0, 1) and transport < 0.35:
            gains[0] *= 0.65
            gains[1] *= 0.65
            res_cost *= 1.25

        # Coupling 3: High communication improves coordinated recovery.
        if action == 3 and comm >= 0.60:
            gains *= 1.20

        # Stage/resource effects (simple, not over-complicated).
        if stage_name == "early" and action == 2:
            gains[2] *= 0.7  # transport-heavy action less effective very early
            invalid_action = True
        if resource_shortage:
            gains *= 0.6
            res_cost *= 1.2
            invalid_action = True

        noise = self.rng.normal(0.0, 0.006, size=3)
        comm = float(np.clip(comm + gains[0] + noise[0], 0.0, 1.0))
        power = float(np.clip(power + gains[1] + noise[1], 0.0, 1.0))
        transport = float(np.clip(transport + gains[2] + noise[2], 0.0, 1.0))

        resources = float(np.clip(resources - res_cost + 0.012, 0.0, 1.0))
        storage = float(np.clip(storage - stor_cost + 0.010, 0.0, 1.0))

        progress = (comm + power + transport) / 3.0
        stage_value, _ = self._stage_from_progress(progress)
        constraint_flag = 1.0 if (resource_shortage or invalid_action) else 0.0
        if constraint_flag > 0.5:
            self.constraint_violation_count += 1

        self.state = np.array([comm, power, transport, resources, storage, stage_value, constraint_flag], dtype=np.float32)
        self.step_count += 1

        progress_delta = float(progress - old_progress)
        reward = 0.45 * progress_delta + 0.25 * power + 0.20 * comm + 0.10 * transport
        if invalid_action:
            reward -= 0.08
        if resource_shortage:
            reward -= 0.07

        terminated = bool(comm >= 0.95 and power >= 0.95 and transport >= 0.92)
        truncated = self.step_count >= self.max_steps
        info = self._build_info(progress_delta=progress_delta, invalid_action=invalid_action, resource_shortage=resource_shortage)
        return self.state.copy(), float(reward), terminated, truncated, info

    def _build_info(self, progress_delta: float, invalid_action: bool, resource_shortage: bool) -> dict[str, Any]:
        comm, power, transport, resources, _storage, stage_value, flag = self.state.tolist()
        stage = "early" if stage_value < 0.3 else "middle" if stage_value < 0.8 else "late"
        return {
            "progress_delta": float(progress_delta),
            "invalid_action": bool(invalid_action),
            "resource_shortage": bool(resource_shortage),
            "stage": stage,
            "constraint_violation": bool(flag > 0.5),
            "constraint_violation_count": int(self.constraint_violation_count),
            "communication_recovery_ratio": float(comm),
            "power_recovery_ratio": float(power),
            "transportation_accessibility_ratio": float(transport),
            "available_repair_resources": float(resources),
        }

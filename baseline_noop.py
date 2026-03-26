"""Baseline no-op revise and intrinsic reward hooks.

Compatible with both old and new call signatures.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def revise_state(state: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
    _ = info
    return np.asarray(state, dtype=float)


def intrinsic_reward(
    state: np.ndarray,
    action: Any = None,
    next_state: np.ndarray | None = None,
    info: dict[str, Any] | None = None,
    revised_state: np.ndarray | None = None,
) -> float:
    _ = (state, action, next_state, info, revised_state)
    return 0.0

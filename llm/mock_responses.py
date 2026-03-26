from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class MockLLMResponseFactory:
    """Deterministic mock response provider for end-to-end testing."""

    seed: int = 0

    def router_response(self) -> str:
        payload = {
            "task_mode": "system_recovery_priority",
            "confidence": 0.86,
            "reason": "Communication and critical load levels are both sub-optimal; balanced recovery is preferred.",
            "stage": "mid_recovery",
        }
        return json.dumps(payload)

    def codegen_response(self, idx: int = 0) -> str:
        module_name = f"generated_candidate_mock_{idx}.py"
        code = f'''from __future__ import annotations
import math
import numpy as np


def _safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


def revise_state(state, info=None):
    s = _safe(state)
    comm, crit, trans = s[0], s[1], s[2]
    progress = float((comm + crit + trans) / 3.0)
    balance_gap = float(abs(comm - crit))
    stage_hint = float(s[5] if s.shape[0] > 5 else 1.0)
    return np.concatenate([s, np.array([progress, balance_gap, stage_hint], dtype=float)], axis=0)


def intrinsic_reward(state, action, next_state, info=None, revised_state=None):
    s = _safe(state)
    ns = _safe(next_state if next_state is not None else state)
    inc = float((ns[0] - s[0]) + 1.2*(ns[1] - s[1]) + 0.8*(ns[2] - s[2]))
    penalty = 0.0
    if info and bool(info.get("constraint_violation", False)):
        penalty += 0.2
    if action == 3:
        inc += 0.01
    return float(math.tanh(2.0 * (inc - penalty)))
'''
        payload = {
            "file_name": module_name,
            "rationale": "Adds progress and balance features and rewards coordinated restoration while penalizing violations.",
            "code": code,
            "expected_behavior": "Improve balanced recovery metrics while avoiding constraint violations.",
        }
        return json.dumps(payload)

    def feedback_response(self) -> str:
        return json.dumps(
            {
                "improvement_focus": "Increase critical load emphasis in early episodes and keep violation penalties.",
                "keep_signals": ["progress", "balance_gap"],
                "avoid_patterns": ["over-penalizing exploration"],
            }
        )

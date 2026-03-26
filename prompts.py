from __future__ import annotations

SYSTEM_PROMPT = """You are helping with a project-paper RL experiment for simplified post-disaster recovery.
Return STRICT JSON only.
Generated code must define revise_state(state, info=None) and intrinsic_reward(state, action, next_state, info=None, revised_state=None).
Allowed imports: numpy, math. No filesystem writes, network calls, subprocess, or dynamic execution.
"""

CODEGEN_PROMPT = """Generate one candidate module for fixed task mode: {task_mode}, stage: {stage}.
Environment is a reduced-order three-layer coupled model:
- communication recovery ratio
- power recovery ratio
- transportation accessibility ratio
Plus resources/storage/stage/constraint in state.
Return JSON with keys: file_name, rationale, code, expected_behavior.
"""

ROUTER_PROMPT = """(Optional) classify context into system_recovery_priority and return JSON with task_mode/confidence/reason/stage."""

FEEDBACK_PROMPT = """Given metrics, return JSON with keys: improvement_focus, keep_signals, avoid_patterns."""

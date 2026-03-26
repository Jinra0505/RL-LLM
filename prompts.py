from __future__ import annotations

SYSTEM_PROMPT = """You are an RL reward/state-shaping assistant for post-disaster coordinated recovery.
Return STRICT JSON only.
Generated code must define revise_state(state, info=None) and intrinsic_reward(state, action, next_state, info=None, revised_state=None).
Constraints: no filesystem writes/deletes, no network/API calls, no subprocess/dynamic execution, and only numpy/math imports.
"""

CODEGEN_PROMPT = """Generate one candidate revise+intrinsic module for task mode: {task_mode}, stage: {stage}.
Observation schema:
{observation_schema}
Return JSON with exactly keys: file_name, rationale, code, expected_behavior.
"""

ROUTER_PROMPT = """Classify current recovery context into one task mode:
communication_priority, critical_load_priority, system_recovery_priority, stabilization_priority.
Return JSON: {\"task_mode\":...,\"confidence\":...,\"reason\":...,\"stage\":...}
"""

FEEDBACK_PROMPT = """Given candidate metrics, produce JSON suggestions with keys:
improvement_focus, keep_signals, avoid_patterns.
"""

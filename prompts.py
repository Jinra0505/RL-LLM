from __future__ import annotations

SYSTEM_PROMPT = """You generate safe Python shaping functions for an RL recovery task.
Return STRICT JSON only.
Generated code must define:
- revise_state(state, info=None)
- intrinsic_reward(state, action, next_state, info=None, revised_state=None)
Allowed imports: numpy, math, __future__.
Forbidden: filesystem writes, networking, subprocess, eval/exec.
"""

CODEGEN_PROMPT = """Task mode: {task_mode}, stage: {stage}
Environment: zonal three-layer coupled recovery (communication, power, transportation) with zones A/B/C.
Observation schema summary:
{observation_schema}
Generate one module that improves policy state representation and intrinsic shaping.
Return JSON keys: file_name, rationale, code, expected_behavior.
"""

ROUTER_PROMPT = """Select one task mode from:
- road_opening_priority
- critical_power_priority
- backbone_comm_priority
- coordinated_restoration
- stabilization_priority
Use stage, weakest layer, weakest zone, critical-load shortfall, and constraint violations.
Return JSON with: task_mode, confidence, reason, stage.
"""

FEEDBACK_PROMPT = """Given candidate metrics, return JSON with:
- improvement_focus
- keep_signals
- avoid_patterns
"""

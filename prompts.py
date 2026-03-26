from __future__ import annotations

SYSTEM_PROMPT = """You generate safe Python shaping functions for a tri-layer recovery RL task.
Return STRICT JSON only.
Generated code must define:
- revise_state(state, info=None)
- intrinsic_reward(state, action, next_state, info=None, revised_state=None)
Allowed imports: numpy, math, __future__.
Forbidden: filesystem writes, networking, subprocess, eval/exec.
Environment state has 24 dims:
0:3 power A/B/C, 3:6 comm A/B/C, 6:9 road A/B/C, 9:12 critical load A/B/C,
12:15 backbone power/comm/road, 15:18 crew power/comm/road, 18 mes location (/2),
19 mes_soc, 20 material_stock, 21 switching_capability, 22 stage_indicator, 23 constraint_flag.
Action space has 14 actions:
0-2 road A/B/C, 3-5 power A/B/C, 6-8 comm A/B/C, 9-11 mes_to A/B/C, 12 feeder_reconfigure, 13 coordinated_balanced.
"""

CODEGEN_PROMPT = """Task mode: {task_mode}, stage: {stage}
Environment: zonal three-layer coupled recovery (communication, power, transportation) with zones A/B/C.
Observation schema summary:
{observation_schema}
Generate one module that improves policy state representation and intrinsic shaping for this 24D state.
Only output code with revise_state and intrinsic_reward (no extra dependencies/modules).
Return JSON keys: file_name, rationale, code, expected_behavior.
"""

ROUTER_PROMPT = """Select one task mode from:
- road_opening_priority
- critical_power_priority
- backbone_comm_priority
- coordinated_restoration
- stabilization_priority
Use stage, weakest layer, weakest zone, critical-load shortfall, backbone_comm_ratio, and violation rates.
Return JSON with: task_mode, confidence, reason, stage.
"""

FEEDBACK_PROMPT = """Given candidate metrics, return JSON with:
- improvement_focus
- keep_signals
- avoid_patterns
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from llm_client import LLMClient


def summarize_trajectory(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    if not trajectory:
        return {"mean_progress_delta": 0.0, "invalid_action_rate": 0.0, "resource_shortage_rate": 0.0, "stage_distribution": {}, "length": 0}
    n = len(trajectory)
    progress = [float(t.get("info", {}).get("progress_delta", 0.0)) for t in trajectory]
    invalid = [1.0 if t.get("info", {}).get("invalid_action", False) else 0.0 for t in trajectory]
    shortage = [1.0 if t.get("info", {}).get("resource_shortage", False) else 0.0 for t in trajectory]
    stages = [t.get("info", {}).get("stage", "unknown") for t in trajectory]
    return {
        "mean_progress_delta": sum(progress) / n,
        "invalid_action_rate": sum(invalid) / n,
        "resource_shortage_rate": sum(shortage) / n,
        "stage_distribution": {k: v / n for k, v in Counter(stages).items()},
        "length": n,
    }


def route_rule(routing_context: dict[str, Any]) -> dict[str, Any]:
    # Router is intentionally lightweight; default paper flow can bypass router.
    env = routing_context.get("env_summary", {})
    traj = routing_context.get("trajectory_summary", {})
    comm = float(env.get("communication_recovery_ratio", 0.0))
    power = float(env.get("power_recovery_ratio", 0.0))
    trans = float(env.get("transportation_accessibility_ratio", 0.0))
    mean_delta = float(traj.get("mean_progress_delta", 0.0))

    avg = (comm + power + trans) / 3.0
    stage = "early" if avg < 0.35 else "middle" if avg < 0.75 else "late"
    if mean_delta < 0.002:
        return {"task_mode": "system_recovery_priority", "confidence": 0.8, "reason": "Progress stalled; balanced recovery needed.", "stage": stage}
    return {"task_mode": "system_recovery_priority", "confidence": 0.75, "reason": "Default fixed task mode for paper setup.", "stage": stage}


def route_llm(client: LLMClient, system_prompt: str, router_prompt: str, routing_context: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": router_prompt + "\n\nContext:\n" + json.dumps(routing_context, indent=2)},
    ]
    route = client.chat_json(messages, response_kind="router")
    for key in ["task_mode", "confidence", "reason", "stage"]:
        if key not in route:
            raise ValueError(f"Router response missing key: {key}")
    return route

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from llm_client import LLMClient


def summarize_trajectory(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    if not trajectory:
        return {
            "mean_progress_delta": 0.0,
            "invalid_action_rate": 0.0,
            "resource_shortage_rate": 0.0,
            "stage_distribution": {},
            "length": 0,
        }

    progress = [float(step.get("info", {}).get("progress_delta", 0.0)) for step in trajectory]
    invalid = [1.0 if step.get("info", {}).get("invalid_action", False) else 0.0 for step in trajectory]
    shortage = [1.0 if step.get("info", {}).get("resource_shortage", False) else 0.0 for step in trajectory]
    stages = [step.get("info", {}).get("stage", "unknown") for step in trajectory]
    stage_counts = Counter(stages)

    n = len(trajectory)
    return {
        "mean_progress_delta": sum(progress) / n,
        "invalid_action_rate": sum(invalid) / n,
        "resource_shortage_rate": sum(shortage) / n,
        "stage_distribution": {k: v / n for k, v in stage_counts.items()},
        "length": n,
    }


def route_rule(routing_context: dict[str, Any]) -> dict[str, Any]:
    env_summary = routing_context.get("env_summary", {})
    traj = routing_context.get("trajectory_summary", {})
    prev = routing_context.get("previous_metrics", {})

    comm = float(env_summary.get("communication_recovery_level", 0.0))
    crit = float(env_summary.get("critical_load_recovery_level", 0.0))
    trans = float(env_summary.get("transportation_accessibility", 0.0))
    violations = int(env_summary.get("constraint_violation_count", 0))

    invalid_rate = float(traj.get("invalid_action_rate", 0.0))
    shortage_rate = float(traj.get("resource_shortage_rate", 0.0))
    mean_delta = float(traj.get("mean_progress_delta", 0.0))
    success_rate = float(prev.get("success_rate", 0.0))

    avg = (comm + crit + trans) / 3.0
    stage = "early_recovery" if avg < 0.35 else "mid_recovery" if avg < 0.75 else "late_recovery"

    if violations > 2 or invalid_rate > 0.2:
        return {"task_mode": "stabilization_priority", "confidence": 0.82, "reason": "High violations/invalid actions.", "stage": stage}
    if shortage_rate > 0.25 or mean_delta < 0.002:
        return {"task_mode": "system_recovery_priority", "confidence": 0.78, "reason": "Progress stalling under shortage.", "stage": stage}
    if crit < comm - 0.08 or crit < 0.45:
        return {"task_mode": "critical_load_priority", "confidence": 0.86, "reason": "Critical load restoration is lagging.", "stage": stage}
    if comm < 0.45 or (success_rate < 0.35 and stage == "early_recovery"):
        return {"task_mode": "communication_priority", "confidence": 0.8, "reason": "Communication restoration is lagging.", "stage": stage}
    return {"task_mode": "system_recovery_priority", "confidence": 0.76, "reason": "Balanced recovery mode.", "stage": stage}


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

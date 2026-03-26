from __future__ import annotations

from typing import Any

from task_router.router import BaseTaskRouter


class RuleTaskRouter(BaseTaskRouter):
    def route(self, summary: dict[str, Any]) -> dict[str, Any]:
        comm = float(summary.get("communication_recovery_level", 0.0))
        crit = float(summary.get("critical_load_recovery_level", 0.0))
        trans = float(summary.get("transportation_accessibility", 0.0))
        violations = int(summary.get("constraint_violation_count", 0))

        if violations > 2:
            return {
                "task_mode": "stabilization_priority",
                "confidence": 0.8,
                "reason": "Constraint violations are elevated; stabilize first.",
                "stage": "late_recovery" if (comm + crit + trans) / 3 > 0.7 else "mid_recovery",
            }
        if crit < 0.45:
            return {
                "task_mode": "critical_load_priority",
                "confidence": 0.87,
                "reason": "Critical load restoration is lagging.",
                "stage": "early_recovery",
            }
        if comm < 0.45:
            return {
                "task_mode": "communication_priority",
                "confidence": 0.84,
                "reason": "Communication restoration is lagging.",
                "stage": "early_recovery",
            }
        return {
            "task_mode": "system_recovery_priority",
            "confidence": 0.82,
            "reason": "Subsystems require balanced improvement.",
            "stage": "mid_recovery" if (comm + crit + trans) / 3 < 0.8 else "late_recovery",
        }

from __future__ import annotations

"""Single-file LLM client with DeepSeek real mode + deterministic mock mode."""

import json
import logging
import os
import time
from typing import Any

LOGGER = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        mode: str = "auto",
        timeout_seconds: int = 45,
        max_retries: int = 2,
        temperature: float = 0.3,
        max_tokens: int = 1400,
    ) -> None:
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        legacy_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.chat_model = os.getenv("DEEPSEEK_MODEL_CHAT", legacy_model)
        self.reasoner_model = os.getenv("DEEPSEEK_MODEL_REASONER", self.chat_model)

        if self.mode == "real" and not self.api_key:
            raise RuntimeError("--llm-mode real selected but DEEPSEEK_API_KEY is not set.")
        if self.mode == "auto" and not self.api_key:
            LOGGER.warning("DEEPSEEK_API_KEY not found, running in mock LLM mode.")

    @property
    def using_mock(self) -> bool:
        return self.mode == "mock" or (self.mode == "auto" and not self.api_key)

    def chat(self, messages: list[dict[str, str]], response_kind: str = "chat", sample_idx: int = 0) -> str:
        if self.using_mock:
            return self._mock_response(response_kind=response_kind, sample_idx=sample_idx)
        return self._real_chat(messages, response_kind=response_kind)

    def chat_json(self, messages: list[dict[str, str]], response_kind: str = "chat", sample_idx: int = 0) -> dict[str, Any]:
        raw = self.chat(messages, response_kind=response_kind, sample_idx=sample_idx)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
            raise ValueError(f"Malformed JSON from LLM ({response_kind}): {raw[:200]}")

    def _mock_response(self, response_kind: str, sample_idx: int) -> str:
        if response_kind == "router":
            task_modes = [
                "road_opening_priority",
                "critical_power_priority",
                "backbone_comm_priority",
                "coordinated_restoration",
                "stabilization_priority",
            ]
            stages = ["early", "middle", "late"]
            return json.dumps(
                {
                    "task_mode": task_modes[sample_idx % len(task_modes)],
                    "confidence": 0.74 + 0.02 * (sample_idx % 6),
                    "reason": "Deterministic mock route for current tri-layer restoration task.",
                    "stage": stages[sample_idx % len(stages)],
                }
            )
        if response_kind == "planning":
            return json.dumps(
                {
                    "weakest_layer": "power",
                    "weakest_zone": "B",
                    "late_stage_risk": "coordinated_overuse_when_prerequisites_weak",
                    "violation_risk": "invalid_mes_or_feeder_actions_under_low_resources",
                    "should_reward": [
                        "weakest-layer targeted gains",
                        "weakest-zone targeted gains",
                        "late-stage low-violation finishing",
                    ],
                    "should_penalize": [
                        "invalid_action",
                        "constraint_violation",
                        "late-stage coordinated action overuse",
                    ],
                    "should_avoid": [
                        "feeder/coordinated actions when backbone_comm, material, or mes_soc are weak",
                        "late-stage road-first detours",
                    ],
                    "finishing_strategy": "close weakest gaps first, then finish with low-violation targeted actions",
                    "codegen_guidance": "Use concise reward terms that prioritize high completion with low violation and discourage risky late-stage coordinated actions.",
                }
            )
        if response_kind == "feedback":
            return json.dumps(
                {
                    "improvement_focus": "Increase critical-load gain and reduce invalid action penalties.",
                    "keep_signals": ["progress_delta", "constraint_violation"],
                    "avoid_patterns": ["over-focusing transport in early recovery"],
                }
            )

        module_name = f"generated_candidate_mock_{sample_idx}.py"
        code = '''from __future__ import annotations
import math
import numpy as np

def _safe(x):
    x = np.asarray(x, dtype=float)
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

def revise_state(state, info=None):
    s = _safe(state)
    p = s[0:3]
    c = s[3:6]
    r = s[6:9]
    l = s[9:12]
    bb = s[12:15]
    mean_power = float(np.mean(p))
    mean_comm = float(np.mean(c))
    mean_road = float(np.mean(r))
    mean_critical = float(np.mean(l))
    backbone_mean = float(np.mean(bb))
    backbone_bottleneck = float(np.min(bb))
    zone_scores = np.array([
        (p[0] + c[0] + r[0] + l[0]) / 4.0,
        (p[1] + c[1] + r[1] + l[1]) / 4.0,
        (p[2] + c[2] + r[2] + l[2]) / 4.0,
    ], dtype=float)
    weakest_zone_score = float(np.min(zone_scores))
    weakest_zone_idx = float(np.argmin(zone_scores))
    critical_load_shortfall = float(max(0.0, 1.0 - mean_critical))
    crew_mean = float(np.mean(s[15:18]))
    material = float(s[20])
    switching = float(s[21])
    mes_soc = float(s[19])
    resource_pressure = float(1.0 - np.clip((crew_mean + material + switching + mes_soc) / 4.0, 0.0, 1.0))
    stage_indicator = float(s[22])
    constraint_flag = float(s[23])
    extras = np.array([
        mean_power,
        mean_comm,
        mean_road,
        mean_critical,
        backbone_mean,
        backbone_bottleneck,
        weakest_zone_score,
        weakest_zone_idx,
        critical_load_shortfall,
        resource_pressure,
        stage_indicator,
        constraint_flag,
    ], dtype=float)
    return np.concatenate([s, extras], axis=0)

def intrinsic_reward(state, action, next_state, info=None, revised_state=None):
    s = _safe(state)
    ns = _safe(next_state if next_state is not None else state)
    dp = float(np.mean(ns[0:3]) - np.mean(s[0:3]))
    dc = float(np.mean(ns[3:6]) - np.mean(s[3:6]))
    dr = float(np.mean(ns[6:9]) - np.mean(s[6:9]))
    dl = float(np.mean(ns[9:12]) - np.mean(s[9:12]))
    base = 0.32 * dp + 0.24 * dc + 0.2 * dr + 0.4 * dl
    zone_before = np.array([
        (s[0] + s[3] + s[6] + s[9]) / 4.0,
        (s[1] + s[4] + s[7] + s[10]) / 4.0,
        (s[2] + s[5] + s[8] + s[11]) / 4.0,
    ], dtype=float)
    zone_after = np.array([
        (ns[0] + ns[3] + ns[6] + ns[9]) / 4.0,
        (ns[1] + ns[4] + ns[7] + ns[10]) / 4.0,
        (ns[2] + ns[5] + ns[8] + ns[11]) / 4.0,
    ], dtype=float)
    weakest_idx = int(np.argmin(zone_before))
    weakest_bonus = float(zone_after[weakest_idx] - zone_before[weakest_idx])
    mes_drop = float(max(0.0, s[19] - ns[19]))
    penalty = 0.0
    if info and bool(info.get("invalid_action", False)):
        penalty += 0.2
    if info and bool(info.get("constraint_violation", False)):
        penalty += 0.25
    penalty += 0.08 * max(0.0, mes_drop - 0.05)
    mode = str((info or {}).get("task_mode", "coordinated_restoration"))
    if mode == "road_opening_priority":
        base += 0.15 * dr
    elif mode == "critical_power_priority":
        base += 0.18 * (dp + dl)
    elif mode == "backbone_comm_priority":
        base += 0.15 * dc
    elif mode == "stabilization_priority":
        penalty += 0.05 * abs(dr + dp + dc)
    total = base + 0.2 * weakest_bonus - penalty
    return float(math.tanh(2.2 * total))
'''
        return json.dumps(
            {
                "file_name": module_name,
                "rationale": "Uses progress+gap shaping and penalizes invalid/violating transitions.",
                "code": code,
                "expected_behavior": "Improve balanced recovery and reduce unsafe actions.",
            }
        )

    def _normalize_base_url(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"

    def _select_model(self, response_kind: str) -> str:
        if response_kind in {"router", "planning", "feedback"}:
            return self.reasoner_model or self.chat_model
        return self.chat_model

    def _real_chat(self, messages: list[dict[str, str]], response_kind: str = "chat") -> str:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "missing openai dependency: install requirements.txt (or `pip install openai>=1.30.0`) to use --llm-mode real."
            ) from exc

        client = OpenAI(api_key=self.api_key, base_url=self._normalize_base_url(), timeout=self.timeout_seconds)
        preferred_model = self._select_model(response_kind=response_kind)
        model_candidates = [preferred_model]
        if preferred_model != self.chat_model:
            model_candidates.append(self.chat_model)

        last_exc: Exception | None = None
        for model in model_candidates:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    return resp.choices[0].message.content or "{}"
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt >= self.max_retries:
                        break
                    LOGGER.warning("DeepSeek call failed (%s/%s, model=%s): %s", attempt + 1, self.max_retries + 1, model, exc)
                    time.sleep(1.5 * (attempt + 1))
            if model != preferred_model:
                LOGGER.warning("Preferred model failed for %s; fallback model succeeded/attempted: %s", response_kind, model)
        raise RuntimeError(f"DeepSeek API call failed after retries: {last_exc}")

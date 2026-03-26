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
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

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
        return self._real_chat(messages)

    def chat_json(self, messages: list[dict[str, str]], response_kind: str = "chat", sample_idx: int = 0) -> dict[str, Any]:
        raw = self.chat(messages, response_kind=response_kind, sample_idx=sample_idx)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
            raise ValueError(f"Malformed JSON from LLM ({response_kind}): {raw[:200]}")

    def _mock_response(self, response_kind: str, sample_idx: int) -> str:
        if response_kind == "router":
            return json.dumps(
                {
                    "task_mode": "coordinated_restoration",
                    "confidence": 0.84,
                    "reason": "Subsystem recovery is uneven; balance restoration priorities.",
                    "stage": "mid_recovery",
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
    progress = float((s[0] + s[1] + s[2]) / 3.0)
    gap = float(abs(s[0] - s[1]))
    stage_hint = float(s[5] if s.shape[0] > 5 else 0.5)
    return np.concatenate([s, np.array([progress, gap, stage_hint], dtype=float)], axis=0)

def intrinsic_reward(state, action, next_state, info=None, revised_state=None):
    s = _safe(state)
    ns = _safe(next_state if next_state is not None else state)
    inc = float((ns[0] - s[0]) + 1.2*(ns[1] - s[1]) + 0.9*(ns[2] - s[2]))
    penalty = 0.0
    if info and bool(info.get("invalid_action", False)):
        penalty += 0.15
    if info and bool(info.get("constraint_violation", False)):
        penalty += 0.2
    if action == 3:
        inc += 0.01
    return float(math.tanh(2.0 * (inc - penalty)))
'''
        return json.dumps(
            {
                "file_name": module_name,
                "rationale": "Uses progress+gap shaping and penalizes invalid/violating transitions.",
                "code": code,
                "expected_behavior": "Improve balanced recovery and reduce unsafe actions.",
            }
        )

    def _real_chat(self, messages: list[dict[str, str]]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=f"{self.base_url.rstrip('/')}/v1", timeout=self.timeout_seconds)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or "{}"
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                LOGGER.warning("DeepSeek call failed (%s/%s): %s", attempt + 1, self.max_retries + 1, exc)
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"DeepSeek API call failed after retries: {last_exc}")

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from llm.mock_responses import MockLLMResponseFactory

LOGGER = logging.getLogger(__name__)


class DeepSeekClient:
    """Single integration point for DeepSeek API access.

    Modes:
    - auto: use real API if key exists, else mock
    - mock: force mock responses
    - real: force real API and fail if key missing
    """

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
        self.mock_factory = MockLLMResponseFactory()

        if self.mode == "real" and not self.api_key:
            raise RuntimeError("--llm-mode real selected but DEEPSEEK_API_KEY is not set.")
        if self.mode == "auto" and not self.api_key:
            LOGGER.warning("DEEPSEEK_API_KEY not found, running in mock LLM mode.")

    @property
    def using_mock(self) -> bool:
        if self.mode == "mock":
            return True
        if self.mode == "auto" and not self.api_key:
            return True
        return False

    def chat(self, messages: list[dict[str, str]], response_kind: str = "chat", sample_idx: int = 0) -> str:
        if self.using_mock:
            if response_kind == "router":
                return self.mock_factory.router_response()
            if response_kind == "codegen":
                return self.mock_factory.codegen_response(sample_idx)
            if response_kind == "feedback":
                return self.mock_factory.feedback_response()
            return json.dumps({"message": "mock response"})
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
                sleep_s = 1.5 * (attempt + 1)
                LOGGER.warning("DeepSeek call failed (attempt %s/%s): %s", attempt + 1, self.max_retries + 1, exc)
                time.sleep(sleep_s)
        raise RuntimeError(f"DeepSeek API call failed after retries: {last_exc}")

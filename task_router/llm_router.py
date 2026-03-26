from __future__ import annotations

from typing import Any

from llm.deepseek_client import DeepSeekClient
from task_router.router import BaseTaskRouter


class LLMTaskRouter(BaseTaskRouter):
    def __init__(self, llm_client: DeepSeekClient, system_prompt: str, router_prompt_template: str) -> None:
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.router_prompt_template = router_prompt_template

    def route(self, summary: dict[str, Any]) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.router_prompt_template + "\n\nContext summary:\n" + str(summary),
            },
        ]
        result = self.llm_client.chat_json(messages, response_kind="router")
        for key in ["task_mode", "confidence", "reason", "stage"]:
            if key not in result:
                raise ValueError(f"Router response missing required key: {key}")
        return result

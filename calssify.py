"""Legacy compatibility shim.

Deprecated misspelled entrypoint retained for backwards compatibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.deepseek_client import DeepSeekClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional LLM-based task routing/classification helper.")
    parser.add_argument("--prompt-file", default="collectdata/prompt_en.txt")
    parser.add_argument("--out-file", default="collectdata/deepseek_result.txt")
    parser.add_argument("--llm-mode", choices=["auto", "mock", "real"], default="auto")
    args = parser.parse_args()

    prompt_path = Path(args.prompt_file)
    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "classify recovery task mode"

    client = DeepSeekClient(mode=args.llm_mode)
    messages = [
        {"role": "system", "content": "You are a precise classifier."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat_json(messages, response_kind="router")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    print(f"Saved classification result to {out_path}")


if __name__ == "__main__":
    main()

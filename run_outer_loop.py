from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from envs.mock_recovery_env import MockRecoveryEnv
from llm.deepseek_client import DeepSeekClient
from task_router.llm_router import LLMTaskRouter
from task_router.rule_router import RuleTaskRouter
from train_rl import run_training

LOGGER = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def safe_parse_codegen_payload(payload: dict[str, Any], fallback_name: str) -> dict[str, str]:
    required = ["file_name", "rationale", "code", "expected_behavior"]
    if not all(k in payload for k in required):
        LOGGER.warning("Malformed codegen payload, applying recovery with fallback skeleton.")
        return {
            "file_name": fallback_name,
            "rationale": "Recovered fallback candidate due to malformed payload.",
            "code": (
                "import numpy as np\n\n"
                "def revise_state(state, info=None):\n"
                "    return np.asarray(state, dtype=float)\n\n"
                "def intrinsic_reward(state, action, next_state, info=None, revised_state=None):\n"
                "    return 0.0\n"
            ),
            "expected_behavior": "Safe no-op fallback.",
        }
    return {k: str(payload[k]) for k in required}


def summarize_env_state() -> dict[str, Any]:
    env = MockRecoveryEnv(max_steps=10, seed=11)
    obs, info = env.reset(seed=11)
    _ = obs
    return {
        "communication_recovery_level": info["communication_recovery_level"],
        "critical_load_recovery_level": info["critical_load_recovery_level"],
        "transportation_accessibility": info["transportation_accessibility"],
        "constraint_violation_count": info["constraint_violation_count"],
    }


def select_best(results: list[dict[str, Any]], metric: str, higher_is_better: bool) -> dict[str, Any]:
    if not results:
        raise ValueError("No candidate results to select from.")
    return sorted(results, key=lambda x: x.get(metric, 0.0), reverse=higher_is_better)[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run LLM outer loop + RL inner loop for recovery tasks.")
    parser.add_argument("--env", default="mock_recovery")
    parser.add_argument("--llm-mode", choices=["auto", "mock", "real"], default="auto")
    parser.add_argument("--router-mode", choices=["off", "rule", "llm"], default="rule")
    parser.add_argument("--fixed-task-mode", default="")
    parser.add_argument("--llm-config", default="configs/llm_config.yaml")
    parser.add_argument("--task-config", default="configs/task_modes.yaml")
    parser.add_argument("--env-schema", default="configs/env_schema.yaml")
    args = parser.parse_args()

    llm_cfg = load_yaml(Path(args.llm_config))
    task_cfg = load_yaml(Path(args.task_config))
    env_schema_cfg = load_yaml(Path(args.env_schema))

    generated_dir = Path(llm_cfg["paths"]["generated_dir"])
    outputs_dir = Path(llm_cfg["paths"]["outputs_dir"])
    generated_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    client = DeepSeekClient(
        mode=args.llm_mode,
        timeout_seconds=int(llm_cfg["llm"]["timeout_seconds"]),
        max_retries=int(llm_cfg["llm"]["max_retries"]),
        temperature=float(llm_cfg["llm"]["temperature"]),
        max_tokens=int(llm_cfg["llm"]["max_tokens"]),
    )

    system_prompt = load_prompt(Path(llm_cfg["prompts"]["system_prompt"]))
    codegen_prompt_template = load_prompt(Path(llm_cfg["prompts"]["codegen_prompt"]))
    router_prompt = load_prompt(Path(llm_cfg["prompts"]["router_prompt"]))
    feedback_prompt = load_prompt(Path(llm_cfg["prompts"]["feedback_prompt"]))

    rounds = int(llm_cfg["outer_loop"]["rounds"])
    candidates_per_round = int(llm_cfg["outer_loop"]["candidates_per_round"])
    score_metric = task_cfg["selection"]["score_metric"]
    higher_is_better = bool(task_cfg["selection"]["higher_is_better"])

    task_mode = args.fixed_task_mode or task_cfg["default_task_mode"]
    stage = "mid_recovery"
    if not args.fixed_task_mode and args.router_mode != "off":
        summary = summarize_env_state()
        if args.router_mode == "rule":
            route = RuleTaskRouter().route(summary)
        else:
            route = LLMTaskRouter(client, system_prompt, router_prompt).route(summary)
        task_mode = route["task_mode"]
        stage = route["stage"]
        LOGGER.info("Router selected task_mode=%s stage=%s confidence=%.2f", task_mode, stage, route["confidence"])

    history: list[dict[str, Any]] = []
    for round_idx in range(rounds):
        LOGGER.info("Starting outer round %s/%s", round_idx + 1, rounds)
        round_candidates: list[dict[str, Any]] = []

        for sample_idx in range(candidates_per_round):
            prompt = codegen_prompt_template.format(
                task_mode=task_mode,
                stage=stage,
                observation_schema=json.dumps(env_schema_cfg["observation_fields"], indent=2),
            )
            if history:
                prompt += "\n\nLatest feedback:\n" + json.dumps(history[-1], indent=2)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            raw_payload = client.chat_json(messages, response_kind="codegen", sample_idx=sample_idx + round_idx * 10)
            candidate = safe_parse_codegen_payload(raw_payload, fallback_name=f"candidate_fallback_r{round_idx}_s{sample_idx}.py")

            candidate_path = generated_dir / candidate["file_name"]
            candidate_path.write_text(candidate["code"], encoding="utf-8")

            result_path = outputs_dir / f"round_{round_idx+1}_candidate_{sample_idx+1}.json"
            metrics = run_training(
                revise_module_path=candidate_path,
                env_name=args.env,
                train_episodes=int(env_schema_cfg["training_defaults"]["train_episodes"]),
                eval_episodes=int(env_schema_cfg["training_defaults"]["eval_episodes"]),
                max_steps_per_episode=int(env_schema_cfg["training_defaults"]["max_steps_per_episode"]),
                gamma=float(env_schema_cfg["training_defaults"]["gamma"]),
                task_mode=task_mode,
                llm_mode="mock" if client.using_mock else "real",
                output_json_path=result_path,
                seed=42 + round_idx * 10 + sample_idx,
            )
            round_candidates.append({
                "candidate": candidate,
                "metrics": metrics,
                "candidate_path": str(candidate_path),
                "result_path": str(result_path),
            })

        best = select_best([c["metrics"] for c in round_candidates], score_metric, higher_is_better)
        summary_payload = {
            "round": round_idx + 1,
            "best_metric": score_metric,
            "best_value": best[score_metric],
            "best_candidate": next(c for c in round_candidates if c["metrics"] is best),
        }
        history.append(summary_payload)

        feedback_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": feedback_prompt + "\n\nCandidate metrics:\n" + json.dumps([c["metrics"] for c in round_candidates], indent=2),
            },
        ]
        feedback = client.chat_json(feedback_messages, response_kind="feedback")
        summary_payload["llm_feedback"] = feedback

        (outputs_dir / f"outer_round_{round_idx+1}_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    final_path = outputs_dir / "outer_loop_final_summary.json"
    final_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    LOGGER.info("Outer loop complete. Final summary: %s", final_path)


if __name__ == "__main__":
    main()

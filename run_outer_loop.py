from __future__ import annotations

import argparse
import ast
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from llm_client import LLMClient
from mock_recovery_env import MockRecoveryEnv
from prompts import CODEGEN_PROMPT, FEEDBACK_PROMPT, ROUTER_PROMPT, SYSTEM_PROMPT
from router import route_llm, route_rule, summarize_trajectory
from train_rl import run_training

LOGGER = logging.getLogger(__name__)
ALLOWED_IMPORTS = {"numpy", "math"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "input"}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_json_with_repair(raw: str) -> tuple[dict[str, Any], bool]:
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(raw[s : e + 1]), True
    return {}, True


def validate_candidate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    required = ["file_name", "rationale", "code", "expected_behavior"]
    for key in required:
        if key not in payload:
            errors.append(f"Missing key: {key}")

    file_name = str(payload.get("file_name", ""))
    code = str(payload.get("code", ""))
    normalized = {
        "file_name": file_name,
        "rationale": str(payload.get("rationale", "")),
        "code": code,
        "expected_behavior": str(payload.get("expected_behavior", "")),
    }

    if not file_name.endswith(".py"):
        errors.append("file_name must end with .py")
    if not code.strip():
        errors.append("code is empty")

    if code.strip():
        try:
            tree = ast.parse(code)
            compile(tree, "<generated>", "exec")
            fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            if "revise_state" not in fn_names:
                errors.append("revise_state not found")
            if "intrinsic_reward" not in fn_names:
                errors.append("intrinsic_reward not found")
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                            errors.append(f"Import not allowed: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    if root not in ALLOWED_IMPORTS:
                        errors.append(f"Import-from not allowed: {node.module}")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_CALLS:
                        errors.append(f"Forbidden call: {node.func.id}")
        except SyntaxError as exc:
            errors.append(f"Syntax error: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Compile error: {exc}")

    return {"valid": len(errors) == 0, "errors": errors, "normalized_payload": normalized}


def collect_routing_context(env_name: str, previous_metrics: dict[str, Any]) -> dict[str, Any]:
    if env_name != "mock_recovery":
        raise ValueError("This minimal prototype supports only mock_recovery")
    env = MockRecoveryEnv(max_steps=20, seed=101)
    state, info = env.reset(seed=101)
    _ = state
    trajectory: list[dict[str, Any]] = []
    for step_idx in range(10):
        action = step_idx % int(env.action_space.n)
        _state, _reward, terminated, truncated, info = env.step(action)
        trajectory.append({"step": step_idx, "action": action, "info": info})
        if terminated or truncated:
            break
    env_summary = {
        "communication_recovery_level": info.get("communication_recovery_level", 0.0),
        "critical_load_recovery_level": info.get("critical_load_recovery_level", 0.0),
        "transportation_accessibility": info.get("transportation_accessibility", 0.0),
        "constraint_violation_count": info.get("constraint_violation_count", 0),
    }
    return {
        "env_summary": env_summary,
        "trajectory_summary": summarize_trajectory(trajectory),
        "previous_metrics": previous_metrics,
    }


def build_feedback(best_candidate: dict[str, Any], score_metric: str) -> dict[str, Any]:
    metrics = best_candidate.get("metrics", {})
    hints: list[str] = []
    if int(metrics.get("constraint_violation_count", 0)) > 4:
        hints.append("Violations are frequent.")
    if float(metrics.get("success_rate", 0.0)) < 0.25:
        hints.append("Success rate is low.")
    if float(metrics.get("mean_progress_delta", 0.0)) < 0.002:
        hints.append("Progress appears stalled.")
    if not hints:
        hints.append("No critical failure detected.")

    return {
        "primary_score_metric": score_metric,
        "primary_score_value": metrics.get(score_metric, 0.0),
        "per_metric_breakdown": metrics,
        "failure_mode_hints": hints,
        "action_usage_summary": metrics.get("action_usage", {}),
        "module_change_summary": {
            "file_name": best_candidate.get("candidate", {}).get("file_name", ""),
            "rationale": best_candidate.get("candidate", {}).get("rationale", ""),
            "expected_behavior": best_candidate.get("candidate", {}).get("expected_behavior", ""),
        },
    }


def select_best(results: list[dict[str, Any]], metric: str, higher_is_better: bool) -> dict[str, Any]:
    return sorted(results, key=lambda x: x.get(metric, 0.0), reverse=higher_is_better)[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Minimal outer loop for LLM-guided recovery shaping.")
    parser.add_argument("--env", default="mock_recovery")
    parser.add_argument("--llm-mode", choices=["auto", "mock", "real"], default="auto")
    parser.add_argument("--router-mode", choices=["off", "rule", "llm"], default="rule")
    parser.add_argument("--fixed-task-mode", default="")
    parser.add_argument("--reroute-each-round", action="store_true")
    parser.add_argument("--rounds-override", type=int, default=0)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    rounds = args.rounds_override or int(cfg["outer_loop"]["rounds"])
    candidates_per_round = int(cfg["outer_loop"]["candidates_per_round"])
    score_metric = str(cfg["selection"]["score_metric"])
    higher_is_better = bool(cfg["selection"]["higher_is_better"])

    generated_dir = Path(cfg["paths"]["generated_dir"])
    outputs_root = Path(cfg["paths"]["outputs_dir"])
    generated_dir.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    client = LLMClient(
        mode=args.llm_mode,
        timeout_seconds=int(cfg["llm"]["timeout_seconds"]),
        max_retries=int(cfg["llm"]["max_retries"]),
        temperature=float(cfg["llm"]["temperature"]),
        max_tokens=int(cfg["llm"]["max_tokens"]),
    )

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = outputs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_snapshot.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "config": cfg,
                "llm_effective_mode": "mock" if client.using_mock else "real",
                "seed_base": 42,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    history: list[dict[str, Any]] = []
    route = {
        "task_mode": args.fixed_task_mode or cfg["task_modes"]["default"],
        "confidence": 0.7,
        "reason": "Initialization",
        "stage": "mid_recovery",
    }

    for round_idx in range(rounds):
        previous_metrics = history[-1].get("best_candidate", {}).get("metrics", {}) if history else {}
        routing_context = collect_routing_context(args.env, previous_metrics)

        if round_idx == 0 or args.reroute_each_round:
            if args.fixed_task_mode:
                route = {"task_mode": args.fixed_task_mode, "confidence": 1.0, "reason": "fixed", "stage": "mid_recovery"}
            elif args.router_mode == "off":
                route = {"task_mode": cfg["task_modes"]["default"], "confidence": 0.7, "reason": "router off", "stage": "mid_recovery"}
            elif args.router_mode == "rule":
                route = route_rule(routing_context)
            else:
                route = route_llm(client, SYSTEM_PROMPT, ROUTER_PROMPT, routing_context)

        round_dir = run_dir / f"round_{round_idx+1}"
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "route.json").write_text(json.dumps(route, indent=2), encoding="utf-8")
        (round_dir / "routing_context.json").write_text(json.dumps(routing_context, indent=2), encoding="utf-8")

        round_candidates: list[dict[str, Any]] = []
        for sample_idx in range(candidates_per_round):
            cid = f"r{round_idx+1}_c{sample_idx+1}"
            cdir = round_dir / cid
            cdir.mkdir(parents=True, exist_ok=True)

            prompt = CODEGEN_PROMPT.format(task_mode=route["task_mode"], stage=route["stage"], observation_schema=json.dumps(cfg["env"]["observation_fields"], indent=2))
            if history:
                prompt += "\n\nLatest feedback:\n" + json.dumps(history[-1].get("feedback_payload", {}), indent=2)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            raw_response = client.chat(messages, response_kind="codegen", sample_idx=sample_idx + round_idx * 10)
            parsed, repaired = parse_json_with_repair(raw_response)
            report = validate_candidate_payload(parsed)
            report["repaired_from_raw"] = repaired

            (cdir / "prompt.txt").write_text(prompt, encoding="utf-8")
            (cdir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
            (cdir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

            record = {"candidate_id": cid, "validation": report, "candidate": report["normalized_payload"]}
            if report["valid"]:
                fname = report["normalized_payload"]["file_name"]
                code = report["normalized_payload"]["code"]
                candidate_path = generated_dir / fname
                candidate_path.write_text(code, encoding="utf-8")
                (cdir / fname).write_text(code, encoding="utf-8")

                metrics = run_training(
                    revise_module_path=candidate_path,
                    env_name=args.env,
                    train_episodes=int(cfg["training"]["train_episodes"]),
                    eval_episodes=int(cfg["training"]["eval_episodes"]),
                    max_steps_per_episode=int(cfg["training"]["max_steps_per_episode"]),
                    gamma=float(cfg["training"]["gamma"]),
                    task_mode=route["task_mode"],
                    llm_mode="mock" if client.using_mock else "real",
                    output_json_path=cdir / "training_result.json",
                    seed=42 + round_idx * 10 + sample_idx,
                )
                record["metrics"] = metrics
                record["candidate_path"] = str(candidate_path)
            else:
                record["metrics"] = {score_metric: -1e9 if higher_is_better else 1e9, "success_rate": 0.0, "constraint_violation_count": 0}
                record["error"] = "Validation failed, skipped training"

            (cdir / "candidate_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            round_candidates.append(record)

        best_metric_values = [c["metrics"] for c in round_candidates]
        best_metrics = select_best(best_metric_values, score_metric, higher_is_better)
        best_candidate = next(c for c in round_candidates if c["metrics"] is best_metrics)

        feedback_payload = build_feedback(best_candidate, score_metric)
        feedback_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": FEEDBACK_PROMPT + "\n\n" + json.dumps(feedback_payload, indent=2)}]
        feedback_raw = client.chat(feedback_messages, response_kind="feedback")
        feedback_json, _ = parse_json_with_repair(feedback_raw)

        summary = {
            "round": round_idx + 1,
            "route": route,
            "best_metric": score_metric,
            "best_value": best_candidate["metrics"].get(score_metric),
            "best_candidate": best_candidate,
            "feedback_payload": feedback_payload,
            "llm_feedback": feedback_json,
            "candidates": round_candidates,
        }
        (round_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        history.append(summary)

    (run_dir / "outer_loop_final_summary.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    LOGGER.info("Outer loop complete. Final summary: %s", run_dir / "outer_loop_final_summary.json")


if __name__ == "__main__":
    main()

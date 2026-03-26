from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from llm_client import LLMClient
from mock_recovery_env import ProjectRecoveryEnv
from prompts import CODEGEN_PROMPT, FEEDBACK_PROMPT, PLANNING_PROMPT, ROUTER_PROMPT, SYSTEM_PROMPT
from router import route_llm, route_rule, summarize_trajectory
from train_rl import run_training

LOGGER = logging.getLogger(__name__)
ALLOWED_IMPORTS = {"numpy", "math", "__future__"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "input"}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_json_with_repair(raw: str) -> tuple[dict[str, Any], bool]:
    try:
        return json.loads(raw), False
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
                return json.loads(cleaned), True
            except json.JSONDecodeError:
                pass
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
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Code validation error: {exc}")

    return {"valid": len(errors) == 0, "errors": errors, "normalized_payload": normalized}


def _action_category(action: int) -> str:
    if action in {0, 1, 2}:
        return "road"
    if action in {3, 4, 5}:
        return "power"
    if action in {6, 7, 8}:
        return "comm"
    if action in {9, 10, 11}:
        return "mes"
    if action == 12:
        return "feeder"
    return "coordinated"


def _aggregate_action_category_distribution(action_usage: dict[str, Any]) -> dict[str, float]:
    cats = {"road": 0.0, "power": 0.0, "comm": 0.0, "mes": 0.0, "feeder": 0.0, "coordinated": 0.0}
    for action_str, val in action_usage.items():
        try:
            action = int(action_str)
            cats[_action_category(action)] += float(val)
        except (TypeError, ValueError):
            continue
    total = sum(cats.values())
    if total > 0.0:
        return {k: v / total for k, v in cats.items()}
    return cats


def _load_revise_fn(module_path: Path | None):
    if not module_path or not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "revise_state", None)
    return fn if callable(fn) else None


def _call_revise(fn: Any, state: Any, info: dict[str, Any]) -> Any:
    if fn is None:
        return state
    try:
        return fn(state, info)
    except TypeError:
        return fn(state)


def _greedy_probe_rollout(env: ProjectRecoveryEnv, revise_fn: Any, horizon: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state, info = env.reset(seed=101)
    trajectory: list[dict[str, Any]] = []
    weakest_zone_freq = {"A": 0, "B": 0, "C": 0}
    for step_idx in range(horizon):
        _ = _call_revise(revise_fn, state, info)
        weakest_zone = str(info.get("weakest_zone", "A"))
        weakest_layer = str(info.get("weakest_layer", "0"))
        zone_to_idx = {"A": 0, "B": 1, "C": 2}
        zone_idx = zone_to_idx.get(weakest_zone, 0)
        weakest_zone_freq[weakest_zone] = weakest_zone_freq.get(weakest_zone, 0) + 1

        if bool(info.get("constraint_violation", False)):
            action = 13
        elif weakest_layer == "2":
            action = zone_idx
        elif weakest_layer == "1":
            action = 6 + zone_idx
        elif weakest_layer == "0":
            action = 3 + zone_idx
        elif float(info.get("mes_soc", 0.0)) > 0.2 and float(info.get("critical_load_shortfall", 1.0)) > 0.3:
            action = 9 + zone_idx
        else:
            action = 13

        next_state, _, terminated, truncated, info = env.step(action)
        trajectory.append({"step": step_idx, "action": action, "info": info})
        state = next_state
        if terminated or truncated:
            break
    return trajectory, weakest_zone_freq


def collect_routing_context(
    env_name: str,
    previous_metrics: dict[str, Any],
    cfg: dict[str, Any],
    previous_best_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if env_name not in {"project_recovery", "mock_recovery"}:
        raise ValueError("Supported env names: project_recovery or mock_recovery")

    enough_previous = all(
        k in previous_metrics
        for k in ["communication_recovery_ratio", "power_recovery_ratio", "road_recovery_ratio", "constraint_violation_count"]
    )
    if enough_previous:
        env_summary = {
            "communication_recovery_ratio": float(previous_metrics.get("communication_recovery_ratio", 0.0)),
            "power_recovery_ratio": float(previous_metrics.get("power_recovery_ratio", 0.0)),
            "road_recovery_ratio": float(previous_metrics.get("road_recovery_ratio", 0.0)),
            "critical_load_shortfall": float(max(0.0, 1.0 - float(previous_metrics.get("critical_load_recovery_ratio", 0.0)))),
            "backbone_comm_ratio": float(previous_metrics.get("backbone_comm_ratio", previous_metrics.get("communication_recovery_ratio", 0.0))),
            "backbone_power_ratio": float(previous_metrics.get("backbone_power_ratio", previous_metrics.get("power_recovery_ratio", 0.0))),
            "backbone_road_ratio": float(previous_metrics.get("backbone_road_ratio", previous_metrics.get("road_recovery_ratio", 0.0))),
            "weakest_zone": str(previous_metrics.get("weakest_zone", "A")),
            "weakest_layer": str(previous_metrics.get("weakest_layer", "0")),
            "constraint_violation_count": int(previous_metrics.get("constraint_violation_count", 0)),
        }
        trajectory_summary = {
            "mean_progress_delta": float(previous_metrics.get("mean_progress_delta", 0.0)),
            "invalid_action_rate": float(previous_metrics.get("invalid_action_rate", 0.0)),
            "constraint_violation_rate": float(previous_metrics.get("constraint_violation_rate", 0.0)),
            "stage_distribution": dict(previous_metrics.get("stage_distribution", {})),
            "action_category_distribution": _aggregate_action_category_distribution(dict(previous_metrics.get("action_usage", {}))),
            "weakest_zone_frequency": dict(previous_metrics.get("weakest_zone_frequency", {})),
            "source": "previous_metrics",
        }
    else:
        env = ProjectRecoveryEnv(
            max_steps=int(cfg["env"].get("max_steps", 60)),
            seed=101,
            severity=str(cfg.get("scenario", {}).get("severity", "moderate")),
            reward_weights=cfg.get("reward_weights", {}),
        )
        module_path = Path(str(previous_best_candidate.get("candidate_path", ""))) if previous_best_candidate else None
        revise_fn = _load_revise_fn(module_path)
        trajectory, weakest_zone_freq = _greedy_probe_rollout(env, revise_fn=revise_fn, horizon=12)
        last_info = trajectory[-1]["info"] if trajectory else {}
        env_summary = {
            "communication_recovery_ratio": float(last_info.get("communication_recovery_ratio", 0.0)),
            "power_recovery_ratio": float(last_info.get("power_recovery_ratio", 0.0)),
            "road_recovery_ratio": float(last_info.get("road_recovery_ratio", 0.0)),
            "critical_load_shortfall": float(last_info.get("critical_load_shortfall", 1.0)),
            "backbone_comm_ratio": float(last_info.get("backbone_comm_ratio", last_info.get("communication_recovery_ratio", 0.0))),
            "backbone_power_ratio": float(last_info.get("backbone_power_ratio", last_info.get("power_recovery_ratio", 0.0))),
            "backbone_road_ratio": float(last_info.get("backbone_road_ratio", last_info.get("road_recovery_ratio", 0.0))),
            "weakest_zone": str(last_info.get("weakest_zone", "A")),
            "weakest_layer": str(last_info.get("weakest_layer", "0")),
            "constraint_violation_count": int(last_info.get("constraint_violation_count", 0)),
        }
        trajectory_summary = summarize_trajectory(trajectory)
        probe_action_usage: dict[str, float] = {}
        for item in trajectory:
            akey = str(item["action"])
            probe_action_usage[akey] = probe_action_usage.get(akey, 0.0) + 1.0
        trajectory_summary["action_category_distribution"] = _aggregate_action_category_distribution(
            probe_action_usage
        )
        trajectory_summary["weakest_zone_frequency"] = weakest_zone_freq
        trajectory_summary["source"] = "greedy_probe_rollout"

    return {
        "env_summary": env_summary,
        "trajectory_summary": trajectory_summary,
        "previous_metrics": previous_metrics,
    }


def build_feedback(best_candidate: dict[str, Any], score_metric: str) -> dict[str, Any]:
    metrics = best_candidate.get("metrics", {})
    hints: list[str] = []
    if int(metrics.get("constraint_violation_count", 0)) > 5:
        hints.append("Constraint violations are frequent.")
    if float(metrics.get("critical_load_recovery_ratio", 0.0)) < 0.6:
        hints.append("Critical load recovery is still low.")
    if float(metrics.get("road_recovery_ratio", 0.0)) < 0.6:
        hints.append("Road restoration is lagging and may bottleneck repairs.")
    if not hints:
        hints.append("No major failure mode detected.")

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


def build_planning_payload(route: dict[str, Any], routing_context: dict[str, Any], previous_feedback: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "task_mode": str(route.get("task_mode", "coordinated_restoration")),
        "stage": str(route.get("stage", "middle")),
        "route_reason": str(route.get("reason", "")),
        "routing_context": routing_context,
        "latest_feedback": previous_feedback or {},
    }


def select_best(results: list[dict[str, Any]], metric: str, higher_is_better: bool) -> dict[str, Any]:
    return sorted(results, key=lambda x: x.get(metric, 0.0), reverse=higher_is_better)[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="LLM outer loop for project-grade tri-layer recovery env.")
    parser.add_argument("--env", default="project_recovery")
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
    higher_is_better = bool(cfg["selection"].get("higher_is_better", True))

    generated_dir = Path(cfg["paths"]["generated_dir"])
    outputs_root = Path(cfg["paths"]["outputs_dir"])
    generated_dir.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    llm_cfg = cfg.get("llm", {})
    if llm_cfg.get("model_chat") and not os.getenv("DEEPSEEK_MODEL_CHAT"):
        os.environ["DEEPSEEK_MODEL_CHAT"] = str(llm_cfg["model_chat"])
    if llm_cfg.get("model_reasoner") and not os.getenv("DEEPSEEK_MODEL_REASONER"):
        os.environ["DEEPSEEK_MODEL_REASONER"] = str(llm_cfg["model_reasoner"])
    if llm_cfg.get("base_url") and not os.getenv("DEEPSEEK_BASE_URL"):
        os.environ["DEEPSEEK_BASE_URL"] = str(llm_cfg["base_url"])

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
        json.dumps({"args": vars(args), "config": cfg, "llm_effective_mode": "mock" if client.using_mock else "real"}, indent=2),
        encoding="utf-8",
    )

    default_mode = str(cfg.get("task_modes", {}).get("default", "coordinated_restoration"))
    history: list[dict[str, Any]] = []
    route = {"task_mode": args.fixed_task_mode or default_mode, "confidence": 0.8, "reason": "default", "stage": "middle"}

    for round_idx in range(rounds):
        previous_best = history[-1].get("best_candidate", {}) if history else None
        prev_metrics = previous_best.get("metrics", {}) if previous_best else {}
        routing_context = collect_routing_context(args.env, prev_metrics, cfg, previous_best_candidate=previous_best)

        if round_idx == 0 or args.reroute_each_round:
            if args.fixed_task_mode:
                route = {"task_mode": args.fixed_task_mode, "confidence": 1.0, "reason": "fixed", "stage": "middle"}
            elif args.router_mode == "off":
                route = {"task_mode": default_mode, "confidence": 0.8, "reason": "router off", "stage": "middle"}
            elif args.router_mode == "rule":
                route = route_rule(routing_context)
            else:
                route = route_llm(client, SYSTEM_PROMPT, ROUTER_PROMPT, routing_context)

        round_dir = run_dir / f"round_{round_idx+1}"
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "route.json").write_text(json.dumps(route, indent=2), encoding="utf-8")
        planning_payload = build_planning_payload(
            route=route,
            routing_context=routing_context,
            previous_feedback=(history[-1].get("llm_feedback", {}) if history else None),
        )
        planning_raw = client.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": PLANNING_PROMPT + "\n\n" + json.dumps(planning_payload, indent=2)}],
            response_kind="planning",
            sample_idx=round_idx,
        )
        planning_json, planning_repaired = parse_json_with_repair(planning_raw)
        (round_dir / "planning_raw.txt").write_text(planning_raw, encoding="utf-8")
        (round_dir / "planning.json").write_text(
            json.dumps({"payload": planning_payload, "planning": planning_json, "repaired_from_raw": planning_repaired}, indent=2),
            encoding="utf-8",
        )

        round_candidates: list[dict[str, Any]] = []
        for sample_idx in range(candidates_per_round):
            cid = f"r{round_idx+1}_c{sample_idx+1}"
            cdir = round_dir / cid
            cdir.mkdir(parents=True, exist_ok=True)

            prompt = CODEGEN_PROMPT.format(
                task_mode=route["task_mode"],
                stage=route["stage"],
                observation_schema=str(cfg["env"]),
                planning_json=json.dumps(planning_json, indent=2, ensure_ascii=False),
            )
            prompt += "\n\nReturn compact JSON and keep generated code concise (<= 80 lines)."
            if history:
                prompt += "\n\nLatest feedback:\n" + json.dumps(history[-1].get("feedback_payload", {}), indent=2)
            raw = "{}"
            parsed: dict[str, Any] = {}
            repaired = True
            report: dict[str, Any] = {"valid": False, "errors": ["not attempted"], "normalized_payload": {}}
            for attempt in range(2):
                attempt_prompt = prompt
                if attempt > 0:
                    attempt_prompt += (
                        "\n\nPrevious output was invalid. Respond with ONLY one JSON object with keys: "
                        "file_name, rationale, code, expected_behavior."
                    )
                raw = client.chat(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": attempt_prompt}],
                    response_kind="codegen",
                    sample_idx=sample_idx + round_idx * 10 + attempt,
                )
                parsed, repaired = parse_json_with_repair(raw)
                report = validate_candidate_payload(parsed)
                report["repaired_from_raw"] = repaired
                if report["valid"]:
                    break

            (cdir / "prompt.txt").write_text(prompt, encoding="utf-8")
            (cdir / "raw_response.txt").write_text(raw, encoding="utf-8")
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
                    max_steps_per_episode=int(cfg["env"]["max_steps"]),
                    gamma=float(cfg["training"]["gamma"]),
                    task_mode=route["task_mode"],
                    llm_mode="mock" if client.using_mock else "real",
                    output_json_path=cdir / "training_result.json",
                    seed=42 + round_idx * 10 + sample_idx,
                    max_revised_dim=(int(cfg.get("state_representation", {}).get("max_revised_dim")) if cfg.get("state_representation", {}).get("max_revised_dim") is not None else None),
                    task_mode_metric_weights=cfg.get("selection", {}).get("task_mode_metric_weights", {}),
                    dqn_cfg=cfg.get("training", {}),
                    severity=str(cfg.get("scenario", {}).get("severity", "moderate")),
                )
                record["metrics"] = metrics
                record["candidate_path"] = str(candidate_path)
                record["task_mode"] = route["task_mode"]
                record["route_source"] = str(routing_context.get("trajectory_summary", {}).get("source", "unknown"))
                record["selection_score"] = float(metrics.get("selection_score", 0.0))
                record["representative_eval_summary"] = dict(metrics.get("representative_eval_summary", {}))
                (cdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            else:
                record["metrics"] = {"selection_score": -1e9 if higher_is_better else 1e9, "success_rate": 0.0}
                record["error"] = "Validation failed"

            (cdir / "candidate_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            round_candidates.append(record)

        best_metrics = select_best([c["metrics"] for c in round_candidates], "selection_score", higher_is_better)
        best_candidate = next(c for c in round_candidates if c["metrics"] is best_metrics)

        feedback_payload = build_feedback(best_candidate, "selection_score")
        feedback_raw = client.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": FEEDBACK_PROMPT + "\n\n" + json.dumps(feedback_payload, indent=2)}], response_kind="feedback")
        feedback_json, _ = parse_json_with_repair(feedback_raw)

        summary = {
            "round": round_idx + 1,
            "route": route,
            "best_metric": "selection_score",
            "best_value": best_candidate["metrics"].get("selection_score"),
            "best_candidate": best_candidate,
            "feedback_payload": feedback_payload,
            "llm_feedback": feedback_json,
        }
        (round_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        history.append(summary)

    (run_dir / "outer_loop_final_summary.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    LOGGER.info("Outer loop complete. Final summary: %s", run_dir / "outer_loop_final_summary.json")


if __name__ == "__main__":
    main()

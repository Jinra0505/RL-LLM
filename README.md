# RL+LLM Tri-Layer Recovery (Current v4 State)

This repository implements a **tri-layer (power/communication/road) restoration environment** with a practical RL+LLM outer loop for shaping-function search.

## 1) Project goal
- Train and evaluate restoration policies in a coupled multi-zone environment.
- Use an LLM outer loop to generate `revise_state` / `intrinsic_reward` modules.
- Keep results auditable via saved routing/planning/candidate/metrics artifacts.

## 2) Environment (tri-layer, 3 zones)
`mock_recovery_env.py::ProjectRecoveryEnv`
- Zones: A/B/C
- Layers: power, communication, road
- Plus critical load, backbone status, crews, MES state, material, switching, stage, constraint flag
- Action space: 14 discrete actions (per-zone repair + MES + feeder + coordinated)

## 3) RL+LLM main flow
Main script: `run_outer_loop.py`

Current real-mode workflow:

`route -> planning(reasoner) -> codegen(chat) -> validate -> train -> select -> feedback`

- Router/planning/feedback requests use reasoner model routing.
- Codegen requests use chat model routing.
- Candidate code must export:
  - `revise_state(state, info=None)`
  - `intrinsic_reward(state, action, next_state, info=None, revised_state=None)`

## 4) LLM modes and model routing
Supported modes:
- `--llm-mode mock`
- `--llm-mode real`
- `--llm-mode auto`

Config keys (`config.yaml`) and env vars are aligned:
- `llm.model_chat` -> `DEEPSEEK_MODEL_CHAT`
- `llm.model_reasoner` -> `DEEPSEEK_MODEL_REASONER`
- `llm.base_url` -> `DEEPSEEK_BASE_URL`

## 5) Repro commands (v4-sized)
Install:
```bash
pip install -r requirements.txt
```

Baseline v4 check:
```bash
python train_rl.py --env project_recovery --llm-mode real --task-mode coordinated_restoration --revise-module baseline_noop.py --config config.yaml --output outputs/exp1_baseline_realcheck_v4.json
```

Real outer-loop v4-style run:
```bash
python run_outer_loop.py --env project_recovery --llm-mode real --router-mode llm --config config.yaml
```

## 6) Formal outputs vs debug artifacts
### Formal outputs (current v4)
- `outputs/exp1_baseline_realcheck_v4.json`
- `outputs/real_outer_loop_v4/outer_loop_final_summary.json`
- `outputs/real_outer_loop_v4/round_1/summary.json`
- `outputs/real_outer_loop_v4/round_1/r1_c1/metrics.json`
- `outputs/real_outer_loop_v4/round_1/r1_c2/metrics.json`
- `outputs/model_diagnosis_v4.md`
- `outputs/model_improvement_report_v4.md`

### Debug / connectivity artifacts
- `outputs/debug_checks/*.json`

These debug files are for SDK/API diagnostics and should not be mixed with formal performance conclusions.

## 7) How to interpret current results (credibility note)
- v4 real-mode pipeline is functional and can achieve strong completion/recovery behavior in many runs.
- Constraint-violation control remains a key optimization target.
- **Important:** v2/v3/v4 success-rate changes are not purely policy-quality effects. Interpretation must account for:
  1. success-rate statistic correction to `eval_success_rate`,
  2. increased `max_steps`,
  3. relaxed done/termination thresholds.

So cross-version conclusions should emphasize **recovery ratios, violation metrics, and late-stage behavior**, not only raw success-rate.

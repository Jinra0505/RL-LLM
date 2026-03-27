# Cleanup Report (Repository State aligned to v4)

## Scope
This cleanup aligns repository structure, documentation, and result references to the current **v4** state.

## 1) Core source files
- `config.yaml`
- `mock_recovery_env.py`
- `train_rl.py`
- `run_outer_loop.py`
- `router.py`
- `llm_client.py`
- `prompts.py`
- `baseline_noop.py`
- `README.md`
- `experiment_summary.md`
- `cleanup_report.md`

## 2) Formal result artifacts (v4)
- `outputs/exp1_baseline_realcheck_v4.json`
- `outputs/real_outer_loop_v4/outer_loop_final_summary.json`
- `outputs/real_outer_loop_v4/round_1/summary.json`
- `outputs/real_outer_loop_v4/round_1/r1_c1/metrics.json`
- `outputs/real_outer_loop_v4/round_1/r1_c2/metrics.json`
- `outputs/model_diagnosis_v4.md`
- `outputs/model_improvement_report_v4.md`

## 3) Debug/check artifacts
Stored separately under:
- `outputs/debug_checks/`

Examples:
- `python_sdk_check.json`
- `python_real_api_check.json`
- `real_api_post_check.json`
- `real_api_router_smoke_test.json`
- `real_api_codegen_smoke_test.json`

## 4) Replaced old references
The following old paths are no longer considered formal current outputs:
- `outputs/exp1_baseline_realcheck.json`
- `outputs/real_outer_loop_smoke/`

All formal documentation now points to v4 result paths.

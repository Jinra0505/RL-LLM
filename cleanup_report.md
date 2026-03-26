# Cleanup Report

## Purpose
Keep runtime artifacts organized so formal experiment outputs are not mixed with ad-hoc connectivity/debug checks.

## Core source files (kept)
- `config.yaml`
- `mock_recovery_env.py`
- `train_rl.py`
- `run_outer_loop.py`
- `router.py`
- `prompts.py`
- `llm_client.py`
- `baseline_noop.py`
- `requirements.txt`
- `README.md`
- `AGENTS.md`

## Formal experiment outputs (kept under `outputs/`)
- `outputs/exp1_baseline_realcheck.json`
- `outputs/real_outer_loop_smoke/outer_loop_final_summary.json`
- `outputs/real_outer_loop_smoke/round_1/summary.json`
- `outputs/real_outer_loop_smoke/round_1/r1_c1/metrics.json`
- `outputs/real_outer_loop_smoke/round_1/r1_c2/metrics.json`

## Debug/check artifacts (moved)
Moved into `outputs/debug_checks/`:
- `python_sdk_check.json`
- `python_real_api_check.json`
- `real_api_post_check.json`
- `real_api_router_smoke_test.json`
- `real_api_codegen_smoke_test.json`

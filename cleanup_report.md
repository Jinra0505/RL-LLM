# Cleanup Report (Current Repository State)

This cleanup focused on repository hygiene and output clarity (no new training runs).

## 1) Core source files retained
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
- `.env.example`
- `.gitignore`
- `AGENTS.md`

## 2) Formal outputs retained
- `outputs/exp1_baseline_realcheck_v4.json`
- `outputs/real_outer_loop_v4/outer_loop_final_summary.json`
- `outputs/real_outer_loop_v4/round_1/summary.json`
- `outputs/real_outer_loop_v4/round_1/r1_c1/metrics.json`
- `outputs/real_outer_loop_v4/round_1/r1_c2/metrics.json`

## 3) Removed stale/debug/duplicate artifacts
- Removed old run directory copy:
  - `outputs/run_20260326_064917/`
- Removed debug API/SDK checks:
  - `outputs/debug_checks/`
- Renamed baseline artifact for consistency:
  - `outputs/exp1_baseline.json` -> `outputs/exp1_baseline_realcheck_v4.json`

## 4) Result organization policy
- Keep only one current formal result set under `outputs/real_outer_loop_v4/`.
- Do not keep duplicated `run_*` snapshots once formal summaries/metrics are extracted.
- Keep transient diagnostics out of formal outputs; if needed later, place under `outputs/debug_checks/` temporarily.

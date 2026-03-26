# RL-LLM Minimal Prototype

This repository is intentionally small and keeps one main pipeline:

`run_outer_loop.py -> router.py -> llm_client.py -> generated candidate -> train_rl.py -> outputs/`

## File layout

- `run_outer_loop.py` : outer-loop orchestration (routing, codegen, validation, selection, feedback)
- `train_rl.py` : RL training/evaluation with dynamic candidate import
- `llm_client.py` : DeepSeek real mode + mock fallback
- `router.py` : trajectory summarization + rule/LLM routing
- `mock_recovery_env.py` : stage-aware mock environment
- `prompts.py` : prompt strings
- `config.yaml` : all runtime config (llm/training/selection/task modes)
- `baseline_noop.py` : fallback revise/reward

## Key behavior

1. Generated code is validated before training (`run_outer_loop.py`).
2. Policy learning uses **revised_state features** directly (not raw-state-only), with optional cap via `state_representation.max_revised_dim` in `config.yaml`.
3. Candidate selection and evaluation summary are **task-mode-dependent** through weighted metrics in `config.yaml` (`selection.task_mode_metric_weights`).

## Install

```bash
pip install -r requirements.txt
```

## Run outer loop (mock mode)

```bash
python run_outer_loop.py --env mock_recovery --llm-mode mock
```

## Run trainer directly

```bash
python train_rl.py --env mock_recovery --llm-mode mock --task-mode system_recovery_priority
```

## Real DeepSeek mode later

```bash
export DEEPSEEK_API_KEY=your_real_key
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat
python run_outer_loop.py --env mock_recovery --llm-mode real
```

## Notes

- Mock mode works end-to-end without any API key.
- This minimal prototype supports only `mock_recovery` environment.
- Outputs are saved under `outputs/run_*` with prompt/response/validation/training artifacts.

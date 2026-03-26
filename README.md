# RL-LLM Minimal Prototype

This repository is intentionally kept small.

Core workflow:

`outer loop -> router -> LLM/mock code generation -> generated revise/reward module -> RL training -> outputs`

## Minimal file layout

- `run_outer_loop.py` : main outer loop
- `train_rl.py` : RL training entrypoint
- `llm_client.py` : DeepSeek (real) + mock client in one file
- `router.py` : trajectory summary + rule/LLM routing in one file
- `mock_recovery_env.py` : runnable Gymnasium mock environment
- `prompts.py` : prompt strings
- `config.yaml` : single config file
- `baseline_noop.py` : fallback revise/reward module
- `generated/` : generated candidate modules
- `outputs/` : run outputs and artifacts

## Install

```bash
pip install -r requirements.txt
```

## Run (mock mode, no API key required)

```bash
python run_outer_loop.py --env mock_recovery --llm-mode mock --router-mode rule --rounds-override 1
```

## Run (auto mode)

```bash
python run_outer_loop.py --env mock_recovery --llm-mode auto --router-mode rule
```

If `DEEPSEEK_API_KEY` is missing, auto mode uses mock mode.

## Run with real DeepSeek later

```bash
export DEEPSEEK_API_KEY=your_real_key
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat
python run_outer_loop.py --env mock_recovery --llm-mode real
```

## Run trainer directly

```bash
python train_rl.py --env mock_recovery --llm-mode mock --task-mode system_recovery_priority --output outputs/rl_result.json
```

## Notes

- Outer loop validates generated code before training (required fields, syntax, imports, function existence).
- This prototype currently supports only `mock_recovery` environment.
- `outputs/run_*` stores prompts, raw responses, validation reports, training results, and round summaries.

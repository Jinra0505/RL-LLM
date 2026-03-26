# RL-LLM Post-Disaster Coordinated Recovery Framework

A runnable backbone for **LLM outer loop + RL inner loop** experiments on coupled communication/power/transport recovery.

## Highlights

- Configuration-driven task modes, prompts, and environment schema.
- Pluggable task router (`rule` or `llm`, or disabled).
- DeepSeek integration via **single entry point**: `llm/deepseek_client.py`.
- Auto mock mode when `DEEPSEEK_API_KEY` is missing.
- End-to-end loop: route -> generate revise/reward code -> train/eval -> feedback -> iterate.
- Generated revise modules saved in `generated/`; experiment outputs saved in `outputs/`.

## Project structure

- `run_outer_loop.py`: outer-loop orchestration.
- `train_rl.py`: RL inner loop trainer with dynamic revise module loading.
- `envs/mock_recovery_env.py`: minimal Gymnasium recovery environment.
- `task_router/`: optional routing subsystem.
- `llm/`: DeepSeek client and structured mock responses.
- `configs/`: task modes, llm settings, environment schema.
- `prompts/`: modular prompt assets.
- `baseline_noop.py`: fallback revise/reward module.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run without API key (explicit mock mode)

```bash
python run_outer_loop.py --env mock_recovery --llm-mode mock
```

## Run with automatic mode

```bash
python run_outer_loop.py --env mock_recovery --llm-mode auto
```

If `DEEPSEEK_API_KEY` is not set, auto mode logs:

> DEEPSEEK_API_KEY not found, running in mock LLM mode.

## Run later with real DeepSeek API

```bash
export DEEPSEEK_API_KEY=your_real_key
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat
python run_outer_loop.py --env mock_recovery --llm-mode real
```

If `--llm-mode real` is used without `DEEPSEEK_API_KEY`, execution fails with a clear error.

## Optional routing modes

```bash
# Rule-based router (default)
python run_outer_loop.py --router-mode rule

# LLM-based router
python run_outer_loop.py --router-mode llm --llm-mode mock

# Disable routing and use configured/fixed mode
python run_outer_loop.py --router-mode off --fixed-task-mode stabilization_priority
```

## Running the RL trainer directly

```bash
python train_rl.py --env mock_recovery --revise-module generated/generated_candidate_mock_0.py --task-mode system_recovery_priority --llm-mode mock
```

If `--revise-module` is omitted, `baseline_noop.py` is used automatically.

## Migration notes from legacy files

Legacy files are preserved (`lesr_main.py`, `lesr_train.py`, etc.) for reference. New entrypoints:

- `lesr_main.py` -> `run_outer_loop.py`
- `lesr_train.py` -> `train_rl.py`
- `calssify.py` remains as compatibility shim; preferred name is `classify.py`

The new framework removes DSO-specific hardcoded assumptions and uses recovery-oriented task modes from config.

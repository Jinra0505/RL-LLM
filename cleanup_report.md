# Cleanup Report

## Removed files/directories
- `.gitkeep` (repository-root placeholder only, not used by the pipeline).

## Why removed
- Not referenced by imports, config, or entrypoints.
- Kept repository focused on RL+LLM tri-layer recovery mainline.

## Core modules retained
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
- `generated/.gitkeep`, `outputs/.gitkeep` (runtime output dirs)

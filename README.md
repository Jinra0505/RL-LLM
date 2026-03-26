# Minimal RL-LLM Project-Paper Prototype

This repository supports exactly one paper-facing comparison in a **reduced-order three-layer coupled recovery environment**:

1. **Experiment 1: RL-Baseline**
2. **Experiment 3: RL + LLM**

Main workflow:

`run_outer_loop.py -> llm_client.py -> generated candidate -> train_rl.py -> outputs/`

## Simplified three-layer coupled environment

`mock_recovery_env.py` models post-disaster recovery with compact aggregated variables (not full network simulation):

- **Communication layer**: communication recovery ratio
- **Power layer**: power/critical-load recovery ratio (+ storage level)
- **Transportation layer**: transportation accessibility ratio

State remains compact (7 dims):

- communication recovery ratio
- power recovery ratio
- transportation accessibility ratio
- available repair resources
- mobile energy storage
- stage indicator (early/middle/late)
- constraint flag

### Coupling rules implemented

1. **Power supports communication**: communication-priority action effectiveness is reduced when power recovery is low.
2. **Transportation supports communication and power restoration**: low transport reduces communication/power action gains and increases resource cost.
3. **Communication improves coordinated recovery**: balanced action gets a gain bonus when communication recovery is high.

## Revised-state policy usage (important)

In `train_rl.py`, policy-state encoding is built from `revised_state` (not raw state only).
By default, all revised features are used; optional cap is controlled by `state_representation.max_revised_dim` in `config.yaml`.

## Task mode usage (paper-simple)

Default flow uses fixed task mode: `system_recovery_priority`.
Router is optional and off by default.
Candidate selection uses task-mode-weighted `selection_score` from `config.yaml`.

## Core metrics

Focused metrics for paper reporting:

- communication recovery ratio
- power recovery ratio
- recovery completion time
- cumulative reward (mean over training episodes)
- constraint violation count

## Install

```bash
pip install -r requirements.txt
```

## Experiment 1: RL-Baseline

```bash
python train_rl.py --env mock_recovery --llm-mode mock --task-mode system_recovery_priority --revise-module baseline_noop.py --output outputs/exp1_baseline.json
```

## Experiment 3: RL + LLM

```bash
python run_outer_loop.py --env mock_recovery --llm-mode mock
```

(Outer loop generates candidate revise/reward code, validates it, trains RL, and saves outputs.)

## Optional real DeepSeek mode

```bash
export DEEPSEEK_API_KEY=your_real_key
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat
python run_outer_loop.py --env mock_recovery --llm-mode real
```

## Notes

- This is intentionally a small project-paper prototype.
- Mock mode works end-to-end without any API key.
- Outputs are written under `outputs/`.

# RL+LLM Tri-Layer Recovery Project (Experiment-Ready)

This repository implements a lightweight but project-grade RL+LLM workflow for post-disaster coordinated recovery:

`run_outer_loop.py -> router.py/llm_client.py -> generated candidate module -> train_rl.py -> outputs`

## Environment: explicit zonal three-layer coupled recovery

`mock_recovery_env.py` now provides `ProjectRecoveryEnv`, a reduced-order but explicit model with:

- 3 zones: **A, B, C**
- 3 layers per zone:
  - power restoration (`P_A,P_B,P_C`)
  - communication restoration (`C_A,C_B,C_C`)
  - road accessibility (`R_A,R_B,R_C`)
- backbone states: `P0, C0, R0`
- critical-load states: `L_A, L_B, L_C`
- process/resource states:
  - crew status (power/comm/road)
  - MES location + SOC
  - material stock
  - switching capability
  - stage indicator
  - constraint flag

Action space: 14 discrete actions
- 0-2 road restoration A/B/C
- 3-5 power restoration A/B/C
- 6-8 communication restoration A/B/C
- 9-11 MES dispatch A/B/C
- 12 feeder reconfiguration / load transfer
- 13 coordinated balanced restoration

## Coupling rules (implemented)

1. **Road -> power/comm repair efficiency** in same zone.
2. **Communication depends on power or MES support** in-zone.
3. **Low C0 reduces feeder/coordinated effectiveness**.
4. **MES dispatch requires road and consumes SOC**.
5. Stage-aware tendencies:
   - early: trunk/backbone and access opening are important,
   - middle: coordinated zonal restoration is efficient,
   - late: stabilization and constraint reduction are favored.

Stage progression uses weighted progress:

`progress = 0.35*mean(power) + 0.35*mean(comm) + 0.30*mean(road)`

## Reward design

`reward = 0.20*delta_power + 0.20*delta_comm + 0.15*delta_road + 0.20*delta_critical_load + 0.10*synergy_bonus - 0.08*constraint_penalty - 0.04*action_switch_penalty - 0.03*mes_overuse_penalty`

## Trainer

`train_rl.py` uses a lightweight DQN (PyTorch):
- MLP Q-network + target network
- replay buffer
- epsilon-greedy
- train/eval split
- JSON output metrics + action usage

`revise_state` really affects policy learning: policy input is built from revised state (`_effective_state`).

## Task modes

Available modes:
- road_opening_priority
- critical_power_priority
- backbone_comm_priority
- coordinated_restoration
- stabilization_priority

Router uses stage + weakest layer + weakest zone + critical-load shortfall + violation frequency.

## Experiments

Install:
```bash
pip install -r requirements.txt
```

### Experiment 1: baseline_noop vs RL+LLM

Baseline (no-op shaping):
```bash
python train_rl.py --env project_recovery --llm-mode mock --task-mode coordinated_restoration --revise-module baseline_noop.py --output outputs/exp1_baseline.json
```

RL+LLM outer loop:
```bash
python run_outer_loop.py --env project_recovery --llm-mode mock
```

### Experiment 3: severity / task-mode comparison

Change severity in `config.yaml` (`mild`, `moderate`, `severe`) and rerun:
```bash
python run_outer_loop.py --env project_recovery --llm-mode mock --router-mode rule
```

You can also pin task mode:
```bash
python run_outer_loop.py --env project_recovery --llm-mode mock --fixed-task-mode backbone_comm_priority
```

## Outputs

- Per-run summaries: `outputs/run_*/outer_loop_final_summary.json`
- Per-candidate metrics: `outputs/run_*/round_*/r*_c*/metrics.json`
- Candidate training detail: `outputs/run_*/round_*/r*_c*/training_result.json`

## Repository status

This repository is cleaned to keep only the RL+LLM tri-layer recovery mainline modules and minimal supporting files.

Optional real DeepSeek:
```bash
export DEEPSEEK_API_KEY=your_real_key
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL_CHAT=deepseek-chat
# llm_client.py reads DEEPSEEK_MODEL; map from *_CHAT if needed
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL_CHAT}"
python run_outer_loop.py --env project_recovery --llm-mode real
```

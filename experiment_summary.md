# Experiment Summary (Real-Mode, Current Repository State)

- Baseline: `outputs/exp1_baseline_realcheck.json`
- Outer-loop final summary: `outputs/real_outer_loop_smoke/outer_loop_final_summary.json`
- Round 1 summary: `outputs/real_outer_loop_smoke/round_1/summary.json`

## Runtime setup used
- mode: real (`--llm-mode real`)
- router: llm (`--router-mode llm`)
- rounds: 1
- candidates_per_round: 2
- training.train_episodes: 140
- training.eval_episodes: 16

## Baseline vs Best Candidate

| metric | baseline | best_candidate |
|---|---:|---:|
| success_rate | 0.0 | 0.0 |
| power_recovery_ratio | 0.7355231530964375 | 0.36023301258683205 |
| communication_recovery_ratio | 0.6257316283881664 | 0.5386956855654716 |
| road_recovery_ratio | 0.6850442122668028 | 0.3700158800929785 |
| critical_load_recovery_ratio | 0.7393128369003534 | 0.36934521794319153 |
| constraint_violation_rate_eval | 0.43645833333333334 | 0.6208333333333332 |
| cumulative_reward_mean | -0.9081026749263255 | 94.58291807390293 |
| selection_score | 0.44893887182697656 | 0.30627544566678505 |

## Route / Validation / Selection
- route task_mode: `backbone_comm_priority`
- best candidate id: `r1_c2`
- best candidate validation: `True`

## Candidate metrics files
- r1_c1: `outputs/real_outer_loop_smoke/round_1/r1_c1/metrics.json`
- r1_c2: `outputs/real_outer_loop_smoke/round_1/r1_c2/metrics.json`

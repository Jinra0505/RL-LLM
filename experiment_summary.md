# Experiment Summary (Current Formal Outputs)

This summary is based on the currently tracked formal artifacts:
- `outputs/exp1_baseline_realcheck_v4.json`
- `outputs/real_outer_loop_v4/outer_loop_final_summary.json`
- `outputs/real_outer_loop_v4/round_1/summary.json`
- `outputs/real_outer_loop_v4/round_1/r1_c1/metrics.json`
- `outputs/real_outer_loop_v4/round_1/r1_c2/metrics.json`

## Baseline v4
- success_rate (eval): **0.0**
- power/comm/road recovery: **0.6211 / 0.6098 / 0.5563**
- critical_load_recovery_ratio: **0.6801**
- constraint_violation_rate_eval: **0.4813**
- selection_score: **0.3976**
- dominant_action_category: **mes**

## Real outer-loop v4
Route:
- task_mode: **critical_power_priority**
- confidence: **0.84**
- stage: **middle**

Best candidate (from `summary.json.best_candidate_id`): **r1_c1**
- r1_c1 selection_score: **0.4660**
- success_rate (eval): **0.0**
- power/comm/road recovery: **0.5775 / 0.4466 / 0.5306**
- critical_load_recovery_ratio: **0.5910**
- constraint_violation_rate_eval: **0.3583**
- dominant_action_category: **mes**

Second candidate:
- r1_c2 selection_score: **0.4052**

## Interpretation
- Current tracked real_v4 run improved violation rate relative to baseline but did not improve success rate (both are 0.0).
- Current tracked baseline outperformed the best real candidate on recovery ratios in this run snapshot.

## Credibility note
Cross-version improvements should not be interpreted from success-rate alone. Historical comparisons are affected by:
1. success-rate metric definition updates,
2. max-step changes,
3. termination-threshold changes.

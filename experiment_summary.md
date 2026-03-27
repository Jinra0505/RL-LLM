# Experiment Summary (Official v4)

## Artifact paths used
- Baseline v4: `outputs/exp1_baseline_realcheck_v4.json`
- Real outer-loop v4 final: `outputs/real_outer_loop_v4/outer_loop_final_summary.json`
- Real round summary: `outputs/real_outer_loop_v4/round_1/summary.json`
- Candidate metrics:
  - `outputs/real_outer_loop_v4/round_1/r1_c1/metrics.json`
  - `outputs/real_outer_loop_v4/round_1/r1_c2/metrics.json`

## v4 key metrics
### Baseline v4
- eval_success_rate: **0.5417**
- power/comm/road recovery: **0.9203 / 0.8688 / 0.9240**
- critical_load_recovery_ratio: **0.9671**
- constraint_violation_rate_eval: **0.5489**
- truncated_rate: **0.4583**
- dominant_action_category: **coordinated**

### Real v4 (best candidate)
- best candidate: **r1_c1**
- route task_mode: **critical_power_priority**
- eval_success_rate: **0.7500**
- power/comm/road recovery: **0.9439 / 0.9111 / 0.8830**
- critical_load_recovery_ratio: **0.9494**
- constraint_violation_rate_eval: **0.5802**
- truncated_rate: **0.2500**
- dominant_action_category: **coordinated**

## Current conclusion
1. Real-mode RL+LLM framework is established and operational in v4.
2. Relative to baseline_v4, real_v4 improves completion and some recovery dimensions.
3. Remaining primary issue is still elevated violation rate and coordinated-heavy late-stage behavior.

## Evaluation credibility note (must-read)
Cross-version (v2/v3/v4) success-rate changes are not purely from strategy improvement.
They are also influenced by protocol/environment changes:
1. success-rate statistic correction to `eval_success_rate`,
2. increased `max_steps`,
3. relaxed done/termination threshold.

Therefore, cross-version interpretation should not rely on success-rate alone.
Recovery ratios, violation metrics, and late-stage behavior remain meaningful comparison axes.

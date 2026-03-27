# Model Improvement Report v4 (Current Tracked Artifacts)

## Scope
This report compares the retained baseline v4 artifact with the retained real outer-loop v4 candidates. No retraining was executed for this update; values are read from existing JSON artifacts.

## Baseline v4
- success_rate: 0.0
- power/comm/road: 0.6211 / 0.6098 / 0.5563
- critical_load_recovery_ratio: 0.6801
- constraint_violation_rate_eval: 0.4812
- selection_score: 0.3976

## Real v4 candidates
### r1_c1
- success_rate: 0.0
- power/comm/road: 0.5775 / 0.4466 / 0.5306
- critical_load_recovery_ratio: 0.5910
- constraint_violation_rate_eval: 0.3583
- selection_score: 0.4660

### r1_c2
- success_rate: 0.0
- power/comm/road: 0.4927 / 0.5414 / 0.4717
- critical_load_recovery_ratio: 0.5297
- constraint_violation_rate_eval: 0.4896
- selection_score: 0.4052

## Interpretation
- In the retained v4 artifacts, best candidate by selection score is r1_c1.
- Baseline remains stronger on recovery ratios, while r1_c1 shows lower violation rate.
- Success rate is 0.0 in all retained artifacts, so conclusions should focus on recovery/violation structure, not completion claims.

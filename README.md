# RL+LLM Tri-Layer Recovery

This repository contains a tri-layer (power/communication/road) recovery environment and an RL+LLM outer-loop search workflow.

## Core workflow
Main entrypoint: `run_outer_loop.py`

Current two-stage LLM loop:

`route -> planning (reasoner) -> codegen (chat) -> validate -> train -> select -> feedback`

- Router/planning/feedback use reasoner routing.
- Code generation uses chat routing.
- Candidate module interface:
  - `revise_state(state, info=None)`
  - `intrinsic_reward(state, action, next_state, info=None, revised_state=None)`

## Core files
- `config.yaml`
- `mock_recovery_env.py`
- `train_rl.py`
- `run_outer_loop.py`
- `router.py`
- `prompts.py`
- `llm_client.py`
- `baseline_noop.py`

## Results layout (clean state)
### Formal tracked outputs
- `outputs/exp1_baseline_realcheck_v4.json`
- `outputs/real_outer_loop_v4/outer_loop_final_summary.json`
- `outputs/real_outer_loop_v4/round_1/summary.json`
- `outputs/real_outer_loop_v4/round_1/r1_c1/metrics.json`
- `outputs/real_outer_loop_v4/round_1/r1_c2/metrics.json`

### Debug artifacts
No debug artifacts are currently tracked. If temporary API/SDK checks are needed, place them under `outputs/debug_checks/` and avoid mixing them with formal results.

## Credibility note for cross-version comparisons
Interpret v2/v3/v4 success changes carefully:
1. success-rate reporting was corrected to eval-based semantics,
2. `max_steps` was increased,
3. done threshold was relaxed.

Therefore, do not attribute all success-rate differences to policy quality alone. Recovery ratios, violation rates, and late-stage behavior remain important comparison axes.

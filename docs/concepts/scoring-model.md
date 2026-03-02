# Scoring Model

## When To Use This Page

Use this page to understand the leaderboard objective and runtime-aware adjustments.

## High-Level Flow

For each budget in the configured budget list:

1. Baseline runtime by depth is measured from sampling.
2. Your estimator streams one depth row at a time.
3. Runtime is checked at each depth against tolerance bounds.
4. Per-depth error is adjusted by runtime behavior.
5. Per-budget adjusted error is aggregated.

Final score is the mean adjusted error across budgets.
Lower is better.

## Runtime Rules

At depth `i`:

- if runtime is above upper bound, that row is zeroed,
- if runtime is below lower bound, effective runtime is floored,
- otherwise measured runtime is used.

This makes timing behavior part of the optimization problem.

## Practical Tuning Intuition

- start with a safe baseline that never violates depth-wise timing,
- add richer logic for larger budgets,
- tune switching behavior empirically with local reports.

## Next

- [Score Report Fields](../reference/score-report-fields.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)

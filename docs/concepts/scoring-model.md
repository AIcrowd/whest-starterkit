# Scoring Model

## 🧠 When to use this page

Use this page to understand how leaderboard score combines estimation quality and compute behavior.

## TL;DR

- Lower score is better.
- Score is based on prediction error plus runtime-aware adjustment.
- You are rewarded for quality under budget, not for unconstrained oversampling.

## High-level flow

For each configured budget:

1. Baseline runtime by depth is measured from sampling.
2. Your estimator streams one depth row at a time.
3. Runtime is checked at each depth against tolerance bounds.
4. Per-depth error is runtime-adjusted.
5. Per-budget adjusted errors are aggregated.

Final score is the mean adjusted error across budgets.

## Runtime rules at depth `i`

- If runtime is above upper bound, that depth row is zeroed.
- If runtime is below lower bound, effective runtime is floored.
- Otherwise measured runtime is used.

This makes budget adaptivity part of the optimization target.

## Why compute-aware scoring

A raw-MSE-only objective can favor strategies that simply spend much more compute on sampling.

This benchmark instead asks a stricter question: can your algorithm produce strong estimates while staying within per-budget runtime expectations?

## Practical tuning intuition

- Start with a safe method that consistently emits valid rows.
- Add richer logic for larger budgets.
- Tune switching behavior using local reports across budgets.

## ➡️ Next step

- [Score Report Fields](../reference/score-report-fields.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)

# Scoring Model

## When to use this page

Use this page to understand how the leaderboard score combines estimation quality and compute behavior.

## TL;DR

- Lower score is better.
- Score reflects prediction accuracy weighted by how much compute you used relative to sampling.
- You are rewarded for quality under budget, not for unconstrained oversampling.

## The core idea

The scoring model answers a specific question: **is your estimator better than just sampling?**

For each compute budget, the evaluator measures how long plain sampling (running random inputs through the circuit) would take. Then it runs your estimator and compares both the accuracy of its predictions and how much time it used.

If your estimator produces the same accuracy as sampling but uses less compute, that's a win. If it produces better accuracy in the same compute, that's also a win. The score captures both dimensions.

## How scoring works

For each configured budget:

1. **Baseline measurement.** The evaluator runs the sampling forward pass at that budget to establish a reference runtime.
2. **Your estimator runs.** Your `predict(circuit, budget)` is called, and the total runtime is measured.
3. **Accuracy is measured.** Per-depth mean squared error (MSE) between your predictions and Monte Carlo ground truth is computed.
4. **Compute is compared.** Your runtime is compared to the sampling baseline. Using more time than sampling incurs a proportional penalty. Using significantly less time does not give unbounded credit — there is a floor.
5. **Score combines both.** Your accuracy is adjusted by your relative compute usage: better accuracy *and* lower compute both push your score down.

Final score is the average adjusted error across all budgets.

## Budget-adaptivity

Your estimator is called at **multiple budgets** (by default: 100, 1,000, 10,000, and 100,000). The same code must handle all of them.

The `budget` argument tells your estimator roughly how many sampling trials would be "allowed" at this level. Smart estimators adapt their strategy:

- At **small budgets** (e.g., 100), sampling is cheap and fast. To compete, your estimator needs to be lightweight too — perhaps just mean propagation.
- At **large budgets** (e.g., 100,000), sampling takes real time. You have room for more sophisticated structural analysis — covariance tracking, iterative refinement, or hybrid methods — because the runtime bar is higher.

The best solutions dynamically allocate their compute across these regimes.

## Runtime rules

The scoring model uses the sampling baseline as a reference clock:

- **Timeout.** If your estimator's total runtime significantly exceeds the sampling baseline (above a tolerance threshold), results incur maximum error.
- **Floor.** If your estimator finishes much faster than sampling, you receive partial but bounded credit — the effective time is clamped to a minimum so that trivially fast (but inaccurate) methods don't game the time ratio.
- **Normal range.** Within the tolerance band around the sampling baseline, your actual runtime is used directly.

The tolerance is configured per evaluation run (typically ±10%).

## What a good score looks like

A score around 1.0 means your estimator performs roughly like sampling — similar accuracy at similar compute. Lower is better.

Scores well below 1.0 mean your structural approach is genuinely beating brute-force sampling: you are getting better predictions per unit of compute. That is exactly the research milestone this challenge targets.

## Practical tuning intuition

- Start with a safe method that consistently emits valid rows.
- Add richer logic for larger budgets.
- Tune switching behavior using local reports across budgets.
- Compare `mse_mean` vs `adjusted_mse` in your reports to diagnose whether runtime or accuracy is the bottleneck.

## ➡️ Next step

- [Score Report Fields](../reference/score-report-fields.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)

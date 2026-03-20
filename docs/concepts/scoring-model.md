# Scoring Model

## When to use this page

Use this page to understand how the leaderboard score combines estimation quality and compute behavior.

## TL;DR

- Lower score is better.
- Score reflects prediction accuracy weighted by how much compute you used relative to sampling.
- You are rewarded for quality under budget, not for unconstrained oversampling.

## The core idea

The scoring model answers a specific question: **is your estimator better than just sampling?**

For each compute budget, the evaluator measures how long plain sampling (running random inputs through the network) would take. Then it runs your estimator and compares both the accuracy of its predictions and how much time it used.

If your estimator produces the same accuracy as sampling but uses less compute, that's a win. If it produces better accuracy in the same compute, that's also a win. The score captures both dimensions.

## How scoring works

For the configured budget:

1. **Baseline measurement.** The evaluator runs the sampling forward pass at that budget to establish a reference runtime.
2. **Your estimator runs.** Your `predict(mlp, budget)` is called. Predictions are collected and the **wall time** is measured.
3. **Accuracy is measured.** Per-depth mean squared error (MSE) between your predictions and Monte Carlo ground truth is computed.
4. **Compute is compared.** Your runtime is compared to the sampling baseline. Using more time incurs a proportional penalty. Using significantly less time does not give unbounded credit — there is a floor.
5. **Score combines both.** Your accuracy is adjusted by your relative compute usage: better accuracy *and* lower compute both push your score down.

Final score is the adjusted error for the budget.

## Budget behavior

Your estimator is called with a `budget` argument. The `budget` tells your estimator roughly how many sampling trials would be "allowed" at this level. Smart estimators adapt their strategy:

- At **small budgets** (e.g., 100), sampling is cheap and fast. To compete, your estimator needs to be lightweight too — perhaps just mean propagation.
- At **large budgets** (e.g., 100,000), sampling takes real time. You have room for more sophisticated structural analysis — covariance tracking, iterative refinement, or hybrid methods — because the runtime bar is higher.

The best solutions dynamically allocate their compute based on budget.

## Runtime rules

The scoring model uses the sampling baseline as a reference clock:

- **Timeout.** If your estimator's total wall time exceeds the sampling baseline time for that MLP, **all** predictions for that MLP are replaced with zeros. This is a hard cutoff, not per-depth.
- **Floor.** If your estimator is much faster than the baseline, the time fraction is clamped to a minimum of 50%. This prevents trivially fast (but inaccurate) methods from gaming the time ratio.
- **Normal range.** Between the floor and timeout, your actual time fraction (`time_spent / time_budget`) is used directly.

The time fraction is computed as `max(time_spent / time_budget, 0.5)`.

## What a good score looks like

The score formula for each MLP is: `final_mse / sampling_mse`, where `sampling_mse = avg_variance / (budget × fraction_spent)`.

A score around 1.0 means your estimator's accuracy is comparable to what sampling would achieve at that budget. Lower is better.

Scores well below 1.0 mean your structural approach is genuinely beating brute-force sampling: you are getting better predictions per unit of compute. That is the research milestone this challenge targets.

## Practical tuning intuition

- Start with a safe method that consistently emits valid rows.
- Add richer logic for larger budgets.
- Tune switching behavior using local reports across budgets.
- Compare `final_mse` vs `primary_score` in your reports to diagnose whether runtime or accuracy is the bottleneck.
- Use [evaluation datasets](../how-to/use-evaluation-datasets.md) to fix networks and ground truth across runs — this makes score comparisons meaningful and skips repeated sampling.

## ➡️ Next step

- [Score Report Fields](../reference/score-report-fields.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)

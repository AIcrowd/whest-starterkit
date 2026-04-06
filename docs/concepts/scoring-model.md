# Scoring Model

## When to use this page

Use this page to understand how the leaderboard score is computed from your estimator's predictions.

## TL;DR

- Lower score is better.
- Score is pure MSE under a FLOP budget constraint.
- If your estimator exceeds the FLOP budget, all predictions for that MLP are zeroed.
- No time normalization, no fraction_spent, no sampling_mse baseline comparison.

## The core idea

The scoring model answers a specific question: **how accurately can your estimator predict expected neuron values within a fixed analytical compute budget?**

Each estimator call is given a `flop_budget` — a cap on the number of floating-point operations it may perform, tracked analytically by mechestim. If the estimator stays within budget, its predictions are scored by MSE against Monte Carlo ground truth. If it exceeds the budget, all predictions for that MLP are replaced with zeros.

## How scoring works

For the configured FLOP budget:

1. **Your estimator runs.** Your `predict(mlp, budget)` is called. mechestim tracks all FLOP usage analytically — no wall-clock measurement.
2. **Budget is checked.** If the total FLOPs used exceed `flop_budget`, all predictions for this MLP are replaced with zero vectors.
3. **Accuracy is measured.** Per-depth mean squared error (MSE) between your predictions and Monte Carlo ground truth is computed.
4. **Score is MSE.** There is no time normalization, no sampling baseline comparison, and no fraction_spent penalty. Your score is the raw MSE of your predictions under budget.

Final score is the MSE averaged across MLPs (zeroed where budget was exceeded).

## Budget behavior

Your estimator receives a `budget` argument (the FLOP budget). It tells your estimator how many FLOPs it may spend in total. Smart estimators adapt their strategy:

- At **small budgets**, only lightweight methods (e.g., mean propagation) fit within the cap. Heavy matrix operations will exceed the budget and zero your predictions.
- At **large budgets**, you have room for more sophisticated structural analysis — covariance tracking, iterative refinement, or hybrid sampling — because the FLOP cap is higher.

The best solutions dynamically allocate their FLOP budget based on the budget value.

## Budget enforcement rules

mechestim enforces the FLOP budget analytically:

- **Exceeded budget.** If your estimator's total FLOPs exceed `flop_budget`, **all** predictions for that MLP are replaced with zeros. This is a hard cutoff, not per-depth.
- **Under budget.** Predictions are used as-is. There is no bonus for using fewer FLOPs than the cap — accuracy is what matters.
- **No floor or clamping.** Unlike the old time-based model, there is no minimum fraction or time floor.

## What a good score looks like

A score near zero means your predictions are highly accurate for those MLPs. A score well above zero means prediction error is high — either because your method is inaccurate, or because it exceeded the FLOP cap and was zeroed.

Scores below what sampling would achieve at that budget indicate your structural approach is genuinely better than brute-force Monte Carlo. That is the research milestone this challenge targets.

## Practical tuning intuition

- Start with a safe method that consistently emits valid rows and stays within budget.
- Use `flop_budget` to gate whether to run more expensive methods.
- Tune switching behavior using local reports across budgets.
- Compare `final_mse` and `all_layer_mse` in your reports to diagnose which depths are hurting your score.
- Use [evaluation datasets](../how-to/use-evaluation-datasets.md) to fix networks and ground truth across runs — this makes score comparisons meaningful and skips repeated sampling.

## Next step

- [Score Report Fields](../reference/score-report-fields.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)

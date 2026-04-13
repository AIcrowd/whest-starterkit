# Scoring Model

## When to use this page

Use this page to understand how the leaderboard score is computed from your estimator's predictions.

## TL;DR

- Lower score is better.
- Score is pure MSE under a FLOP budget constraint.
- If your estimator exceeds the FLOP budget, all predictions for that MLP are zeroed.
- Your score is pure MSE — the closer to zero, the better.

## The core idea

The scoring model answers a specific question: **how accurately can your estimator predict expected neuron values within a fixed analytical compute budget?**

Each estimator call is given a `flop_budget` — a cap on the number of floating-point operations it may perform, tracked analytically by mechestim. If the estimator stays within budget, its predictions are scored by MSE against Monte Carlo ground truth. If it exceeds the budget, all predictions for that MLP are replaced with zeros.

## How scoring works

For the configured FLOP budget:

1. **Your estimator runs.** Your `predict(mlp, budget)` is called. mechestim tracks all FLOP usage analytically — no wall-clock measurement.
2. **Budget is checked.** If the total FLOPs used exceed `flop_budget`, all predictions for this MLP are replaced with zero vectors.
3. **Accuracy is measured.** Per-depth mean squared error (MSE) between your predictions and Monte Carlo ground truth is computed.
4. **Score is MSE.** Your score is pure MSE — the closer to zero, the better.

Final score is the MSE averaged across MLPs (zeroed where budget was exceeded).

## Budget behavior

Your estimator receives a `budget` argument (the FLOP budget). You may use it to route between cheap and expensive algorithms — the combined estimator example does this. But you are not required to. Fixed-strategy estimators that always use the same approach work fine, as long as they stay within budget.

## Budget enforcement rules

mechestim enforces the FLOP budget analytically:

- **Exceeded budget.** If your estimator's total FLOPs exceed `flop_budget`, **all** predictions for that MLP are replaced with zeros. This is a hard cutoff, not per-depth.
- **Under budget.** Predictions are used as-is. There is no bonus for using fewer FLOPs than the cap — accuracy is what matters.
- **No floor or clamping.** Predictions are scored as-is — there is no minimum fraction or penalty floor.

## What a good score looks like

A score near zero means your predictions are highly accurate for those MLPs. A score well above zero means prediction error is high — either because your method is inaccurate, or because it exceeded the FLOP cap and was zeroed.

Scores below what sampling would achieve at that budget indicate your structural approach is genuinely better than brute-force Monte Carlo. That is the research milestone this challenge targets.

## Practical tuning intuition

- Start with a safe method that consistently emits valid rows and stays within budget.
- Use `flop_budget` to gate whether to run more expensive methods.
- Tune switching behavior using local reports across budgets.
- Compare `final_mse` and `all_layer_mse` in your reports to diagnose which depths are hurting your score.
- Use [evaluation datasets](../how-to/use-evaluation-datasets.md) to fix networks and ground truth across runs — this makes score comparisons meaningful and skips repeated sampling.

## Worked example

Suppose ground truth for a 3-neuron final layer is `[0.42, 0.38, 0.51]` and your estimator predicts `[0.40, 0.35, 0.55]`.

    MSE = mean((0.40 - 0.42)^2 + (0.35 - 0.38)^2 + (0.55 - 0.51)^2)
        = mean(0.0004 + 0.0009 + 0.0016)
        = 0.000967

Your `primary_score` for this MLP would be approximately 0.001. Across all MLPs in the evaluation, scores are averaged to produce your final ranking.

## Example estimator benchmarks

The table below shows real scores from the four bundled example estimators, run with default settings (width=100, depth=16, 10 MLPs, 100M FLOP budget). Use these as calibration points for your own estimator.

| Estimator | Final MSE | All-Layer MSE | Approach |
|-----------|-----------|---------------|----------|
| `random_estimator` | ~0.50 | ~0.48 | Returns zeros — the baseline. Any structural method should beat this. |
| `mean_propagation` | ~0.004 | ~0.002 | Diagonal variance, O(depth x width^2). ~100x better than baseline. |
| `covariance_propagation` | ~0.0003 | ~0.0002 | Full covariance, O(depth x width^3). ~1000x better than baseline. |
| `combined_estimator` | ~0.0002 | ~0.0002 | Routes to covariance when budget allows, otherwise mean propagation. |

**How to read these numbers:**

- The **random estimator** (zeros) gives you the "doing nothing" baseline. Its MSE reflects the natural scale of the ground truth activations.
- **Mean propagation** is ~100x more accurate than zeros — a huge improvement from a simple analytical formula with negligible FLOP cost.
- **Covariance propagation** is another ~10x better, but costs O(width^3) per layer. At width=100, this is affordable; at width=1000, it would exhaust the budget.
- The **combined estimator** gets the best of both worlds by checking the budget before deciding which algorithm to run.

To reproduce: `whest run --estimator examples/estimators/<name>.py --n-mlps 10`

Scores vary slightly between runs due to random MLP generation and Monte Carlo ground truth noise.

## Next step

- [Score Report Fields](../reference/score-report-fields.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)

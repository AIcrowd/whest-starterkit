# Frequently Asked Questions

## Can I use numpy directly?

All computation must go through mechestim (`import mechestim as me`). mechestim wraps numpy with analytical FLOP counting — your score depends on the FLOP cost of your operations, and only mechestim tracks those costs.

## Can I use scipy?

Yes. scipy is not part of mechestim, so you import it separately as your own dependency. Common usage: `scipy.special.ndtr` for the standard normal CDF. Add `scipy` to your `requirements.txt` when packaging.

## Why is my score `inf`?

Your estimator raised an exception or returned invalid data (wrong shape, NaN, non-numeric). The framework treats that MLP as a failure. Run with `--debug` to see the error:

```bash
nestim run --estimator ./my-estimator/estimator.py --runner local --debug
```

## Do I need to use the `budget` argument in `predict()`?

The `budget` argument tells you how many FLOPs you are allowed. You can use it to choose between cheap and expensive algorithms (like the combined estimator does), or you can ignore it and always use the same strategy.

mechestim enforces the budget regardless — if your operations exceed it, `BudgetExhaustedError` is raised and your predictions are zeroed.

## Can I precompute things in `setup()`?

Yes. `setup()` runs before any `predict()` calls and is not under a FLOP budget. Use it for one-time preparation that does not depend on the specific MLP (e.g., lookup tables, configuration).

However, `setup()` does have a time limit (`setup_timeout_s`, typically 5 seconds).

## What happens if I exceed the FLOP budget?

mechestim raises `BudgetExhaustedError` before the over-budget operation executes. The framework catches this and zeros all your predictions for that MLP. You will see `budget_exhausted: true` in the per-MLP report.

## Is scoring hardware-dependent?

No. mechestim counts FLOPs analytically based on tensor shapes — not wall-clock time. The same estimator produces the same FLOP count on any hardware. You can develop on a laptop and submit for evaluation on a cluster with identical results.

## How many MLP networks are in a full evaluation?

The default evaluation scores your estimator on 10 MLPs (configured by `n_mlps` in `ContestSpec`). Each MLP has the same width and depth but different random weights. Your aggregate score is the mean MSE across all MLPs.

## What if my estimator is fast but inaccurate?

You are ranked by MSE, not by how few FLOPs you use. Using fewer FLOPs than the budget gives no bonus — only accuracy matters (as long as you stay within budget).

## Next step

- [Common Participant Errors](./common-participant-errors.md)
- [Debugging Checklist](../how-to/debugging-checklist.md)
- [Scoring Model](../concepts/scoring-model.md)

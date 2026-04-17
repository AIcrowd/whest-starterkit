# Frequently Asked Questions

## Can I use numpy directly?

All computation must go through whest (`import whest as we`). whest wraps numpy with analytical FLOP counting — your score depends on the FLOP cost of your operations, and only whest tracks those costs.

## Can I use scipy?

Yes. scipy is not part of whest, so you import it separately as your own dependency. Common usage: `scipy.special.ndtr` for the standard normal CDF. Add `scipy` to your `requirements.txt` when packaging.

## Why is my score `inf`?

Either your estimator raised during `predict()`, returned invalid data (wrong shape, NaN, non-numeric), or exhausted the FLOP / wall-time budget. Since 2026-04, explicit predict errors surface through an "Estimator Errors" panel in the report and set exit code `1`; `inf` with exit `0` means budget or time exhaustion rather than a swallowed exception. Run with `--debug` to see the tracebacks:

```bash
whest run --estimator ./my-estimator/estimator.py --debug
whest run --estimator ./my-estimator/estimator.py --debug --fail-fast   # halt at first error
```

## Do I need to use the `budget` argument in `predict()`?

The `budget` argument tells you how many FLOPs you are allowed. You can use it to choose between cheap and expensive algorithms (like the combined estimator does), or you can ignore it and always use the same strategy.

whest enforces the budget regardless — if your operations exceed it, `BudgetExhaustedError` is raised and your predictions are zeroed.

## Can I precompute things in `setup()`?

Yes. `setup()` runs before any `predict()` calls and is not under a FLOP budget. Use it for one-time preparation that does not depend on the specific MLP (e.g., lookup tables, configuration).

However, `setup()` does have a time limit (`setup_timeout_s`, typically 5 seconds).

## How do I set a time limit on my estimator code?

At the whest level, time limits live on `BudgetContext` via
`wall_time_limit_s`:

```python
with we.BudgetContext(flop_budget=10_000_000, wall_time_limit_s=2.0) as budget:
    ...
```

In WhestBench CLI runs, `--wall-time-limit` sets that same limit for each
`predict()` call.

## What is `untracked_time_limit`?

`untracked_time_limit` is a WhestBench rule, not a `BudgetContext`
parameter. Whest reports:

- `tracked_time_s`: time spent inside counted whest calls
- `untracked_time_s`: total wall time minus tracked time

WhestBench can then zero predictions if `untracked_time_s` exceeds the
configured `--untracked-time-limit`.

## What happens if I exceed the FLOP budget?

whest raises `BudgetExhaustedError` before the over-budget operation executes. The framework catches this and zeros all your predictions for that MLP. You will see `budget_exhausted: true` in the per-MLP report.

## How do I inspect budget summaries while debugging?

Use:

- `budget.summary()` for the current explicit `BudgetContext`
- `we.budget_summary()` for the accumulated process/session view
- `budget.summary_dict(...)` or `we.budget_summary_dict(...)` for structured data

If you want namespace attribution, pass `by_namespace=True`.

## Is scoring hardware-dependent?

No. whest counts FLOPs analytically based on tensor shapes — not wall-clock time. The same estimator produces the same FLOP count on any hardware. You can develop on a laptop and submit for evaluation on a cluster with identical results.

## How many MLP networks are in a full evaluation?

The default evaluation scores your estimator on 10 MLPs (configured by `n_mlps` in `ContestSpec`). Each MLP has the same width and depth but different random weights. Your aggregate score is the mean MSE across all MLPs.

## What if my estimator is fast but inaccurate?

You are ranked by MSE, not by how few FLOPs you use. Using fewer FLOPs than the budget gives no bonus — only accuracy matters (as long as you stay within budget).

## Next step

- [Common Participant Errors](./common-participant-errors.md)
- [Debugging Checklist](../how-to/debugging-checklist.md)
- [Scoring Model](../concepts/scoring-model.md)

# Common Participant Errors

Use this page when `validate` or `run` fails.

## Understand runner modes first

`nestim run --estimator ...` uses `--runner server` by default.

- `server` (default): realistic isolation — your estimator runs against the mechestim server.
- `local`: in-process execution with best traceback fidelity while debugging.

Fast debug ladder:

```bash
nestim run --estimator ./my-estimator/estimator.py
nestim run --estimator ./my-estimator/estimator.py --debug
nestim run --estimator ./my-estimator/estimator.py --runner local --debug
```

Sample server-style failure:

```text
Error [setup:SETUP_ERROR]: Estimator setup failed.
Use --debug to include a traceback.
Tip: For estimator-level tracebacks, rerun with --runner local --debug.
```

Exact follow-up:

```bash
nestim run --estimator ./my-estimator/estimator.py --runner local --debug
```

## Estimator returned wrong shape

Symptom: error mentions expected shape `(depth, width)`.

Why it happens: returned wrong dimensions or a 1D array.

Fix now: ensure `predict` returns a mechestim array with shape `(mlp.depth, mlp.width)`. Use `me.zeros((mlp.depth, mlp.width))` as a starting point.

Verify:

```bash
nestim validate --estimator ./my-estimator/estimator.py
```

## Non-finite values (`nan` or `inf`)

Symptom: error mentions finite values.

Why it happens: unstable numeric operations.

Fix now: add guards/clipping/checks in your prediction logic.

Verify:

```bash
nestim validate --estimator ./my-estimator/estimator.py
```

## FLOP budget exceeded

Symptom: unexpectedly poor `primary_score` despite reasonable prediction logic.

Why it happens: your estimator exceeded the FLOP budget, causing all predictions for that MLP to be zeroed.

Fix now:

- check `flops_used` and `budget_exhausted` in the per-MLP report,
- reduce expensive operations (matmul dominates FLOP cost),
- consider diagonal approximations instead of full covariance,
- see [Manage Your FLOP Budget](../how-to/manage-flop-budget.md) for optimization guidance.

Verify:

```bash
nestim run --estimator ./my-estimator/estimator.py --runner server --json
```

## Next step

- [Estimator Contract](../reference/estimator-contract.md)
- [Scoring Model](../concepts/scoring-model.md)

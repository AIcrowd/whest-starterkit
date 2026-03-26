# Common Participant Errors

Use this page when `validate` or `run` fails.

## Understand runner modes first

`nestim run --estimator ...` uses `--runner subprocess` by default.

- `subprocess`: realistic isolation and safer runtime boundary.
- `inprocess`: best local traceback fidelity while debugging your estimator.

Fast debug ladder:

```bash
nestim run --estimator ./my-estimator/estimator.py
nestim run --estimator ./my-estimator/estimator.py --debug
nestim run --estimator ./my-estimator/estimator.py --runner inprocess --debug
```

Sample subprocess-style failure:

```text
Error [setup:SETUP_ERROR]: Estimator setup failed.
Use --debug to include a traceback.
Tip: For estimator-level tracebacks, rerun with --runner inprocess --debug.
```

Exact follow-up:

```bash
nestim run --estimator ./my-estimator/estimator.py --runner inprocess --debug
```

## Estimator returned wrong shape

Symptom: error mentions expected shape `(depth, width)`.

Why it happens: returned wrong dimensions or a 1D array.

Fix now: ensure `predict` returns an `np.ndarray` with shape `(mlp.depth, mlp.width)`.

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

## Runtime envelope penalties

Symptom: unexpectedly poor `primary_score` despite reasonable `final_mse`.

Why it happens: estimator exceeding the time budget, causing predictions to be zeroed.

Fix now:

- simplify compute for smaller budgets,
- keep total runtime under the sampling baseline,
- compare `final_mse` vs `primary_score` to diagnose timing penalties.

Verify:

```bash
nestim run --estimator ./my-estimator/estimator.py --runner subprocess --json
```

## Next step

- [Estimator Contract](../reference/estimator-contract.md)
- [Scoring Model](../concepts/scoring-model.md)

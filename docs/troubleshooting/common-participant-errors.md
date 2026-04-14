# Common Participant Errors

Use this page when `validate` or `run` fails.

## Understand runner modes first

`whest run --estimator ...` uses `--runner server` by default.

- `server` (default): realistic isolation — your estimator runs against the whest server.
- `local`: in-process execution with best traceback fidelity while debugging.

Fast debug ladder:

```bash
whest run --estimator ./my-estimator/estimator.py
whest run --estimator ./my-estimator/estimator.py --debug
whest run --estimator ./my-estimator/estimator.py --runner local --debug
```

Sample server-style failure:

```text
Error [setup:SETUP_ERROR]: Estimator setup failed.
Use --debug to include a traceback.
Tip: For estimator-level tracebacks, rerun with --runner local --debug.
```

Exact follow-up:

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --debug
```

## Estimator returned wrong shape

Symptom: error mentions expected shape `(depth, width)`.

Why it happens: returned wrong dimensions or a 1D array.

Fix now: ensure `predict` returns a whest array with shape `(mlp.depth, mlp.width)`. Use `we.zeros((mlp.depth, mlp.width))` as a starting point.

Verify:

```bash
whest validate --estimator ./my-estimator/estimator.py
```

## Non-finite values (`nan` or `inf`)

Symptom: error mentions finite values.

Why it happens: unstable numeric operations.

Fix now: add guards/clipping/checks in your prediction logic.

Verify:

```bash
whest validate --estimator ./my-estimator/estimator.py
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
whest run --estimator ./my-estimator/estimator.py --runner server --json
```

## Class not found

Symptom: "No estimator class found" or `ImportError`.

Why it happens: your class must be named `Estimator` (or specify `--class`).

Fix now: rename your class to `Estimator`.

Verify:

```bash
whest validate --estimator ./my-estimator/estimator.py
```

## Import error in estimator

Symptom: `ModuleNotFoundError` when loading your file.

Why it happens: your estimator imports a module not installed in the environment.

Fix now: add missing dependencies to `requirements.txt`. For whest, use `import whest as we`.

Verify:

```bash
whest validate --estimator ./my-estimator/estimator.py
```

## Signature mismatch

Symptom: `TypeError: predict() missing 1 required positional argument`.

Why it happens: your `predict` method has the wrong signature.

Fix now: ensure signature is `def predict(self, mlp: MLP, budget: int) -> we.ndarray:`.

Verify:

```bash
whest validate --estimator ./my-estimator/estimator.py
```

## Setup timeout

Symptom: `SETUP_TIMEOUT` error.

Why it happens: `setup()` exceeded the time limit (typically 5 seconds).

Fix now: move expensive computation from `setup()` to `predict()`, or reduce setup work.

Verify:

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --debug
```

## Predict timeout

Symptom: `PREDICT_TIMEOUT` error.

Why it happens: `predict()` exceeded the wall-clock safety limit.

Fix now: check for infinite loops or extremely expensive operations. This is a safety guardrail, not the FLOP budget.

Verify:

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --debug
```

## Budget exhausted mid-operation

Symptom: `BudgetExhaustedError` raised during a specific operation.

Why it happens: a single whest operation would exceed your remaining FLOP budget.

Fix now: use `we.budget_summary()` to find the expensive operation. Consider diagonal approximations or fewer iterations.

Verify: check `flops_used` in the score report.

## Numerical instability in deep networks

Symptom: predictions become `nan` or `inf` after many layers.

Why it happens: values grow or shrink exponentially through deep networks without safeguards.

Fix now: add overflow guards — rescale covariance when diagonal values exceed a threshold (see `covariance_propagation.py` example). Use float64 for intermediate calculations.

Verify:

```bash
whest validate --estimator ./my-estimator/estimator.py
```

## Dtype mismatch

Symptom: output is float64 but evaluator expects float32, or similar type issues.

Why it happens: whest operations may produce different dtypes than expected.

Fix now: cast your output: `return we.asarray(result, dtype=we.float32)`.

Verify:

```bash
whest validate --estimator ./my-estimator/estimator.py
```

## Empty predictions

Symptom: returned shape `(0, width)` or similar zero-length array.

Why it happens: your layer loop did not iterate (empty `mlp.weights`).

Fix now: check that you iterate over `mlp.weights` and append results per layer.

Verify:

```bash
whest validate --estimator ./my-estimator/estimator.py
```

## Using numpy instead of whest

Symptom: operations work but FLOP budget is not consumed (shows 0 flops_used).

Why it happens: you are using `import numpy as np` instead of `import whest as we`. Numpy operations are not FLOP-tracked.

Fix now: replace all `np.*` calls with `we.*` equivalents. See [Code Patterns](../reference/code-patterns.md).

Verify: check `flops_used > 0` in score report.

## Score is inf

Symptom: `primary_score` shows as `inf`.

Why it happens: your estimator raised an exception during `predict()`.

Fix now: run with `--runner local --debug` to see the full traceback.

Verify:

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --debug
```

## Setup runs expensive operations

Symptom: unexpected FLOP usage or budget consumption before `predict()`.

Why it happens: `setup()` runs outside any `BudgetContext`, so whest operations there use the default (very large) budget. This is fine — but if you accidentally do heavy computation in setup that should be in predict, you lose budget awareness.

Fix now: keep `setup()` lightweight. Move estimation logic to `predict()`.

## Next step

- [Debugging Checklist](../how-to/debugging-checklist.md)
- [FAQ](./faq.md)
- [Estimator Contract](../reference/estimator-contract.md)
- [Scoring Model](../concepts/scoring-model.md)

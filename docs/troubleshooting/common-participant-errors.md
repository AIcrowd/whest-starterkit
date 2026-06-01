# Common Participant Errors

> [← Documentation](../README.md)

Use this page when `validate` or `run` fails.

## Understand runner modes first

`whest run --estimator ...` uses `--runner local` by default.

- `local` (default): in-process execution with best traceback fidelity while debugging.
- `subprocess`: isolated process execution for stricter reproduction; `server` remains a legacy alias.

Fast debug ladder:

```bash
whest run --estimator estimator.py
whest run --estimator estimator.py --debug
whest run --estimator estimator.py --runner local --debug
```

Sample server-style failure:

```text
Error [setup:SETUP_ERROR]: Estimator setup failed.
Use --debug to include a traceback.
Tip: For estimator-level tracebacks, rerun with --runner local --debug.
```

Exact follow-up:

```bash
whest run --estimator estimator.py --runner local --debug
```

## Estimator returned wrong shape

Symptom: error mentions expected shape `(depth, width)`.

Why it happens: returned wrong dimensions or a 1D array.

Fix now: ensure `predict` returns a flopscope array with shape `(mlp.depth, mlp.width)`. Use `fnp.zeros((mlp.depth, mlp.width))` as a starting point.

Verify:

```bash
whest validate --estimator estimator.py
```

## Non-finite values (`nan` or `inf`)

Symptom: error mentions finite values.

Why it happens: unstable numeric operations.

Fix now: add guards/clipping/checks in your prediction logic.

Verify:

```bash
whest validate --estimator estimator.py
```

## FLOP budget exceeded

Symptom: unexpectedly poor `adjusted_final_layer_score` despite reasonable prediction logic, with one or more MLPs showing `budget_exhausted: true` or `combined_budget_exhausted: true`.

Why it happens: your estimator's effective compute `C_m = F_m + λ·R_m` exceeded `flop_budget`. The affected MLP's predictions are replaced with zeros and the per-MLP multiplier is forced to **1.0** (no compute discount), so `adjusted_final_layer_score_m = MSE(0, Y_m) × 1.0` — strictly worse than a trivial-zero submission that succeeds (which gets the 0.1 multiplier floor).

`budget_exhausted` fires when flopscope itself trips (your analytical FLOPs exceed the cap). `combined_budget_exhausted` fires on the post-hoc check `C_m > B` — flopscope didn't trip, but the residual-wall-time penalty (`λ · residual_wall_time_s` at `λ = 1e11` FLOPs/sec) pushed effective compute past the cap.

Fix now:

- check `flops_used`, `effective_compute`, `residual_wall_time_s`, `budget_exhausted`, and `combined_budget_exhausted` in the per-MLP report,
- reduce expensive operations (matmul dominates FLOP cost),
- reduce Python-side overhead — tight loops over neurons add to `residual_wall_time_s` and thus to `effective_compute`,
- consider diagonal approximations instead of full covariance,
- see [Manage Your FLOP Budget](../how-to/manage-flop-budget.md) for optimization guidance.

Verify:

```bash
whest run --estimator estimator.py --json
```

## Class not found

Symptom: "No estimator class found" or `ImportError`.

Why it happens: your class must be named `Estimator` (or specify `--class`).

Fix now: rename your class to `Estimator`.

Verify:

```bash
whest validate --estimator estimator.py
```

## Import error in estimator

Symptom: `ModuleNotFoundError` when loading your file.

Why it happens: your estimator imports a module not installed in the environment.

Fix now: add missing dependencies to `requirements.txt`. For flopscope, use `import flopscope as flops` and `import flopscope.numpy as fnp`.

Verify:

```bash
whest validate --estimator estimator.py
```

## Signature mismatch

Symptom: `TypeError: predict() missing 1 required positional argument`.

Why it happens: your `predict` method has the wrong signature.

Fix now: ensure signature is `def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:`.

Verify:

```bash
whest validate --estimator estimator.py
```

## Predict raised an unexpected exception

Symptom: `whest run` exits with status `1` and prints an "Estimator Errors" panel listing one or more MLPs with a `PREDICT_ERROR` code. A stderr line reads e.g. `2 of 10 MLP(s) raised during predict; rerun with --debug for tracebacks...`.

Why it happens: your `predict()` raised an exception that is neither `BudgetExhaustedError` nor `TimeExhaustedError`. WhestBench routes the failure through the zero-prediction path — the affected MLP scores `final_layer_mse_m × 1.0` (no compute discount) and the suite mean stays finite. The non-zero exit code signals that the submission is not yet passing.

Fix now:

```bash
# Show full tracebacks in the "Estimator Errors" panel:
whest run --estimator estimator.py --debug

# Stop at the first failure and propagate the raw Python traceback:
whest run --estimator estimator.py --debug --fail-fast
```

The traceback in the panel (or the raw stack from `--fail-fast`) points directly at the line in your estimator that raised.

## Setup timeout

Symptom: `SETUP_TIMEOUT` error.

Why it happens: `setup()` exceeded the time limit (typically 5 seconds).

Fix now: move expensive computation from `setup()` to `predict()`, or reduce setup work.

Verify:

```bash
whest run --estimator estimator.py --runner local --debug
```

## Predict timeout

Symptom: `PREDICT_TIMEOUT` error.

Why it happens: `predict()` exceeded the wall-clock safety limit.

Fix now: check for infinite loops or extremely expensive operations. This is a safety guardrail, not the FLOP budget.

Verify:

```bash
whest run --estimator estimator.py --runner local --debug
```

## Budget exhausted mid-operation

Symptom: `BudgetExhaustedError` raised during a specific operation.

Why it happens: a single flopscope operation would exceed your remaining FLOP budget.

Fix now: use `flops.budget_summary()` to find the expensive operation. Consider diagonal approximations or fewer iterations.

Verify: check `flops_used` in the score report.

## Numerical instability in deep networks

Symptom: predictions become `nan` or `inf` after many layers.

Why it happens: values grow or shrink exponentially through deep networks without safeguards.

Fix now: add overflow guards — rescale covariance when diagonal values exceed a threshold (see `covariance_propagation.py` example). Use float64 for intermediate calculations.

Verify:

```bash
whest validate --estimator estimator.py
```

## Dtype mismatch

Symptom: output is float64 but evaluator expects float32, or similar type issues.

Why it happens: flopscope operations may produce different dtypes than expected.

Fix now: cast your output: `return fnp.asarray(result, dtype=fnp.float32)`.

Verify:

```bash
whest validate --estimator estimator.py
```

## Empty predictions

Symptom: returned shape `(0, width)` or similar zero-length array.

Why it happens: your layer loop did not iterate (empty `mlp.weights`).

Fix now: check that you iterate over `mlp.weights` and append results per layer.

Verify:

```bash
whest validate --estimator estimator.py
```

## Using numpy instead of flopscope

Symptom: operations work but FLOP budget is not consumed (shows 0 flops_used).

Why it happens: you are using `import numpy as np` instead of `import flopscope.numpy as fnp`. Numpy operations are not FLOP-tracked.

Fix now: replace all `np.*` calls with `fnp.*` equivalents. See [Code Patterns](../reference/code-patterns.md).

Verify: check `flops_used > 0` in score report.

## Every MLP failed (n_failed_mlps == n_mlps)

Symptom: the suite-level `failure_breakdown` shows every MLP carrying at least one failure flag (`n_failed_mlps == n_mlps`), and the `adjusted_final_layer_score` is dominated by `MSE(0, Y_m) × 1.0` across the board (typically lands near `0.5`, the raw `final_layer_mse` of zero predictions at the default activation scale).

> **Note:** this is the post-merge replacement for the older "score is `inf`" symptom. Since whestbench PR #39 (May 2026) failures produce finite scores at the zero-prediction × 1.0 multiplier — there is no longer an `inf` path.

Why it happens: every MLP either raised during `predict()` or exhausted the FLOP / wall-time / residual-wall-time / combined budget.

Tell them apart from `failure_breakdown` and the exit code:

- **Exit `1` + non-zero `failure_breakdown.error` + "Estimator Errors" panel** — `predict()` raised exceptions on at least one MLP. See [Predict raised an unexpected exception](#predict-raised-an-unexpected-exception).
- **Exit `0` + every `per_mlp[i].budget_exhausted: true`** — you ran out of analytical FLOPs.
- **Exit `0` + every `per_mlp[i].combined_budget_exhausted: true`** — your `effective_compute = F_m + λ·R_m` exceeded the cap (residual wall time pushed you over, even though flopscope didn't trip).
- **Exit `0` + every `per_mlp[i].time_exhausted: true`** — you ran out of wall-clock time.

Fix now: run with `--debug` to see tracebacks in the "Estimator Errors" panel (works with any runner), or `--fail-fast` to halt at the first failing MLP with the raw Python stack:

```bash
whest run --estimator estimator.py --debug
whest run --estimator estimator.py --debug --fail-fast
```

## Setup runs expensive operations

Symptom: unexpected FLOP usage or budget consumption before `predict()`.

Why it happens: `setup()` runs outside any `BudgetContext`, so flopscope operations there use the default (very large) budget. This is fine — but if you accidentally do heavy computation in setup that should be in predict, you lose budget awareness.

Fix now: keep `setup()` lightweight. Move estimation logic to `predict()`.

## ➡️ Next step

- [Debugging Checklist](../how-to/debugging-checklist.md)
- [FAQ](./faq.md)
- [Estimator Contract](../reference/estimator-contract.md)
- [Scoring Model](../concepts/scoring-model.md)

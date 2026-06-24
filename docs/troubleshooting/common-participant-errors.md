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

> **Local runs use the full flopscope; the grader uses the flopscope *client*.** `whest run` (both `--runner local` and `--runner subprocess`) executes against the full, locally-installed `flopscope` package, while the grader runs the lighter flopscope *client* — a numpy-compatible proxy. The two are designed to match, so the single best habit is to write all array code against `flopscope.numpy` (`import flopscope.numpy as fnp`) and never reach for plain numpy. Be aware, though, that a few client-only parity gaps can pass locally and surface **only** in grading — the local runner does not exercise the client/server split. Most of the failures documented below are exactly those gaps; when the grader reports one, the fix is on this page.

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

`budget_exhausted` fires when flopscope itself trips (your analytical FLOPs exceed the cap). `combined_budget_exhausted` fires on the post-hoc check `C_m > B` — flopscope didn't trip, but the residual-wall-time penalty (`λ · residual_wall_time_s`, λ default `1e11` FLOPs/sec) pushed effective compute past the cap.

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

## Result array too large

Symptom: on the **grading server** (not local runs), a per-MLP failure whose message reads `result array too large: N bytes exceeds 4294967296 byte limit` — or `array too large: …` for an input you build. It can also fail your submission at the **smoke test**, before grading starts.

Why it happens: the flopscope runtime on the grading sandbox caps any single array at **4 GiB** (a memory-safety guard). It applies to both arrays you build via `flopscope.numpy` and the array you return from `predict()`. A single array that large almost always means an over-vectorized "all layers × all samples at once" buffer, or `float64` where the MLP weights are `float32`.

Note: local `whest run` uses an in-process backend **without** this cap, so you won't reproduce it locally — keep your peak single-array size under 4 GiB as a rule.

Fix now:

- process samples/rows/columns in **blocks** and accumulate running statistics instead of materializing one giant array,
- keep arrays `float32` (a stray `float64` buffer is 2× larger),
- reshapes and allocations cost **0 FLOP**, so chunking is free against your FLOP budget.

## Class not found

Symptom: "No estimator class found" or `ImportError`.

Why it happens: your class must be named `Estimator` (or specify `--class`).

Fix now: rename your class to `Estimator`.

Verify:

```bash
whest validate --estimator estimator.py
```

## Packaging / submission rejected (`IMPORT_FAILED`)

Symptom: your submission is rejected before any MLP runs, with a message such as `IMPORT_FAILED`, a `manifest.json` schema / `api_version` error, `'estimator.py' sha256 mismatch`, or "tarball missing manifest.json".

Why it happens: the grader unpacks your archive and checks it against the `manifest.json` it expects — entrypoint, declared versions, and a SHA-256 for every file. A hand-rolled or hand-edited archive (wrong layout, a file changed after the manifest was generated, a missing or stale manifest) fails this gate, so nothing is graded.

Fix now: never assemble the tarball yourself. Re-run the provided packaging command so the manifest and file hashes are generated together and stay in sync:

```bash
# single-file estimator
uv run whest package --estimator estimator.py --output submission.tar.gz

# multi-file estimator (weights, helper modules) — point at the folder
uv run whest package --estimator . --output submission.tar.gz
```

`whest` prints exactly what it bundled before writing the archive; if a file you expected isn't listed, fix that *before* submitting. See [Stage 5: Package Your Submission](../getting-started/stage-5-package.md) and the [Pre-Submission Checklist](../how-to/pre-submission-checklist.md).

Verify:

```bash
tar tf submission.tar.gz   # should list estimator.py (+ your files) and manifest.json
```

## Import error in estimator

Symptom: `ModuleNotFoundError` when loading your file.

Why it happens: your estimator imports something the grader sandbox doesn't provide. At grading time only `flopscope` (incl. `flopscope.numpy as fnp`), the `whestbench` API (`BaseEstimator`, `MLP`, `SetupContext`), and the Python standard library are importable — there is **no `requirements.txt` install step**, so third-party packages (`numpy`, `scipy`, `torch`, …) are not available. (A missing *helper module* is different: it ships if you package the folder — `--estimator .` — instead of the single file.)

Fix now: for all array math use `import flopscope as flops` and `import flopscope.numpy as fnp` (not `import numpy`). Ship multi-file estimators as a folder. For work that genuinely needs a third-party library (a PyTorch-trained model, a scipy routine), compute it **offline** before packaging and ship the result as a pickle-free `.npz`, loaded in `setup()` — see [Ship Weights](../how-to/ship-weights.md). Note: `whest validate` runs in your local venv (which *has* numpy/scipy/torch), so it will **not** reproduce a grader-only missing-package error — the only fix is to not import those packages.

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

## Setup failed (`SETUP_ERROR` / `SETUP_FAILED`)

Symptom: locally, `whest run` prints `Error [setup:SETUP_ERROR]: Estimator setup failed.`; on the grader it is reported as `SETUP_FAILED: <Exception>`. Either way, the submission is rejected before any MLP is scored.

Why it happens: your estimator's `setup()` raised. Unlike a `predict()` failure (which is isolated to a single MLP and zero-scored), an exception in `setup()` rejects the **whole** submission — there is nothing to grade if setup never completes. Common causes: a weights/`.npz` file that didn't ship or loads with the wrong path, an assertion or config read that's brittle on the grader, or work that's fine locally but trips on a sandbox difference.

Fix now: make `setup()` exception-proof. Load files via the path the framework gives you, guard fallible work defensively (try/except with a sane fallback), and keep it lightweight — heavy work belongs in `predict()` (it also avoids the [setup timeout](#setup-timeout)). Reproduce locally with the isolated runner before submitting:

```bash
whest run --estimator estimator.py --runner subprocess --debug
```

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

Why it happens: you are using `import numpy as np` instead of `import flopscope.numpy as fnp`. Numpy operations are not FLOP-tracked. (This is the *local* symptom — your venv has numpy, so it runs silently untracked. On the grader `import numpy` fails outright, since the sandbox has no numpy — see [Import error in estimator](#import-error-in-estimator). Either way, `np.*` is wrong; use `fnp.*`.)

Fix now: replace all `np.*` calls with `fnp.*` equivalents. See [Code Patterns](../reference/code-patterns.md).

Verify: check `flops_used > 0` in score report.

## No numpy in the sandbox (`No module named 'numpy'` / `name 'np' is not defined`)

Symptom: on the grader, `ModuleNotFoundError: No module named 'numpy'` — or, if your `import numpy as np` was wrapped in a `try`/`except` that swallowed it, a later `NameError: name 'np' is not defined` at the first `np.*` call.

Why it happens: the grading sandbox does **not** ship raw numpy. It provides `flopscope.numpy` (a FLOP-counting, numpy-compatible proxy) instead. `import numpy` simply has nothing to import, so it raises — and a swallowed import leaves `np` undefined, which surfaces as a `NameError` further down. (See also the broader [Import error in estimator](#import-error-in-estimator) for the full list of what the sandbox does and doesn't provide.)

Fix now: import the flopscope array module and write all array code against it:

```python
import flopscope.numpy as fnp     # or: import flopscope.numpy as np
```

Bind it to `np` if you don't want to rename every call site — but do not `import numpy` and do not silently `except ImportError` around it.

Verify:

```bash
whest run --estimator estimator.py --runner subprocess
```

## flopscope arrays are immutable

Symptom: `flopscope arrays are immutable, so item assignment ... is not supported`, or `in-place add (arr += x) is not supported` (and similar for other in-place operators).

Why it happens: arrays produced by `flopscope.numpy` are read-only proxies. Mutating ops — `arr[i] = v`, `arr[mask] = v`, `arr += x`, `arr *= x` — are rejected by design, on the grader and (with the client) locally.

Fix now: build new arrays functionally instead of mutating in place:

- replace `arr[i] = v` / `arr[mask] = v` with `fnp.where(mask, v, arr)`, or rebuild via slicing + `fnp.concatenate`,
- replace `arr += x` with `arr = arr + x` (and likewise `*=`, `-=`, `/=`),
- set a diagonal with `fnp.fill_diagonal(M, v)` (cheap — `min(m, n)` FLOP) instead of indexed assignment.

See [Code Patterns](../reference/code-patterns.md) for the functional equivalents.

Verify:

```bash
whest run --estimator estimator.py --runner subprocess
```

## Reduction `axis` must be an int or tuple, not a list

Symptom: `TypeError: 'list' object cannot be interpreted as an integer` from a reduction such as `sum`, `mean`, `max`, or `prod`.

Why it happens: numpy itself only accepts `axis=<int>`, `axis=<tuple>`, or `axis=None` for reductions — a `list` axis is rejected. flopscope mirrors numpy exactly, so passing a list fails the same way it would in plain numpy.

Fix now: pass a tuple, not a list:

```python
a.sum(axis=(0, 1))     # correct
a.sum(axis=[0, 1])     # TypeError
```

Verify:

```bash
whest run --estimator estimator.py --runner local --debug
```

## Every MLP failed (n_failed_mlps == n_mlps)

Symptom: the suite-level `failure_breakdown` shows every MLP carrying at least one failure flag (`n_failed_mlps == n_mlps`), and the `adjusted_final_layer_score` is dominated by `MSE(0, Y_m) × 1.0` across the board (typically lands near `0.91`, the raw `final_layer_mse` of zero predictions at the default activation scale).

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

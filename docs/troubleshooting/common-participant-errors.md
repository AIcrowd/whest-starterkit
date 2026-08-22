# Common participant errors

> [← Documentation](../README.md)

Use this page when `validate` or `run` fails.

## Understand runner modes first

`whest run --estimator ...` uses `--runner local` by default.

- `local` (default): in-process execution with best traceback fidelity while debugging.
- `subprocess`: isolated process execution for stricter reproduction; `server` remains a legacy alias.

Fast debug ladder:

```bash
uv run whest run --estimator estimator.py
uv run whest run --estimator estimator.py --debug
uv run whest run --estimator estimator.py --runner local --debug
```

Sample server-style failure:

```text
Error [setup:SETUP_ERROR]: weights.npz not found
Use --debug to include a traceback.
Tip: For estimator-level tracebacks, rerun with --runner local --debug.
```

The text after `SETUP_ERROR:` is your own exception's message, so it will not
match this example; `[setup:SETUP_ERROR]` is the part to recognize.

Exact follow-up:

```bash
uv run whest run --estimator estimator.py --runner local --debug
```

> **Local runs use the full flopscope; the grader uses the flopscope *client*.** `whest run` (both `--runner local` and `--runner subprocess`) executes against the full, locally-installed `flopscope` package, while the grader runs the lighter flopscope *client*, a numpy-compatible proxy. The two are designed to match, so the single best habit is to write all array code against `flopscope.numpy` (`import flopscope.numpy as fnp`) and never use plain numpy. A few client-only parity gaps can still pass locally and surface **only** in grading; the local runner does not exercise the client/server split. Most of the failures documented below are exactly those gaps; when the grader reports one, the fix is on this page.

## Estimator returned wrong shape

Symptom: error mentions expected shape `(depth, width)`.

Why it happens: returned wrong dimensions or a 1D array.

Fix now: ensure `predict` returns a flopscope array with shape `(mlp.depth, mlp.width)`. Use `fnp.zeros((mlp.depth, mlp.width))` as a starting point.

Verify:

```bash
uv run whest validate --estimator estimator.py
```

## Non-finite values (`nan` or `inf`)

Symptom: error mentions finite values.

Why it happens: unstable numeric operations.

Fix now: add guards/clipping/checks in your prediction logic.

Verify:

```bash
uv run whest validate --estimator estimator.py
```

## FLOP budget exceeded

Symptom: unexpectedly poor `adjusted_final_layer_score` despite reasonable prediction logic, with one or more MLPs showing `budget_exhausted: true` or `combined_budget_exhausted: true`.

Why it happens: your estimator's effective compute exceeded `flop_budget`. In Phase 2 that is your analytical FLOP count, `C_m = F_m`, against a per-MLP budget of `2**41` (2,199,023,255,552 FLOPs). The affected MLP's predictions are replaced with zeros and the per-MLP multiplier is forced to **1.0** (no compute discount), so `adjusted_final_layer_score_m = MSE(0, Y_m) × 1.0` — strictly worse than a trivial-zero submission that succeeds (which gets the 0.1 multiplier floor).

`budget_exhausted` fires when flopscope itself trips (the operation about to run would exceed the cap). `combined_budget_exhausted` fires on the post-hoc check `C_m > B` after `predict()` returns; because Phase 2 scores `C_m = F_m`, that is a backstop on the same FLOP count rather than a second way to fail. Residual wall time is no longer priced into `C_m`; it has its own 400 ms cap, reported as `residual_wall_time_exhausted` (next section).

Fix now:

- check `flops_used`, `effective_compute`, `residual_wall_time_s`, `budget_exhausted`, and `combined_budget_exhausted` in the per-MLP report,
- reduce expensive operations (matmul dominates FLOP cost),
- reduce Python-side overhead; tight loops over neurons add to `residual_wall_time_s`, which has its own hard cap,
- consider diagonal approximations instead of full covariance,
- see [Manage Your FLOP Budget](../how-to/manage-flop-budget.md) for optimization guidance.

Verify:

```bash
uv run whest run --estimator estimator.py --json
```

If you have profiled an operation and believe flopscope is billing it wrongly, send a minimal repro (op, shapes, dtypes, the `ctx.summary()` rows) to [arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com) rather than budgeting around it.

## Residual wall-time cap exceeded

Symptom: a per-MLP entry with `residual_wall_time_exhausted: true` — locally, a `ResidualWallTimeExhaustionWarning` naming the MLP and the measured time. Predictions for that MLP are zeroed and the multiplier is forced to **1.0**.

Why it happens: everything that is not a metered flopscope call (your Python control flow, per-neuron loops, GC, I/O) accumulates in `residual_wall_time_s`, and it is capped at **400 ms per MLP**. The cap is not priced: unused FLOPs do not extend it, and going over is a failure rather than a surcharge.

That 400 ms is sized for **plumbing**: control flow, slicing inputs into blocks, bookkeeping between flopscope calls. It is not an allowance for computation. Doing meaningful compute outside flopscope (raw Python arithmetic over neurons, a vendored array library, a compiled kernel, threads, or a subprocess) is prohibited and disqualifiable, whether or not it fits inside the cap. See [Allowed Code](../concepts/allowed-code.md) for the full boundary.

Fix now:

- move array work into `flopscope.numpy`, where it is metered analytically instead of timed,
- replace per-neuron / per-layer Python loops with whole-array ops (one `(depth, width)` array, not `depth × width` scalar steps),
- move MLP-independent work into `setup()`, whose wall time is not charged to the residual at all,
- if you have a legitimate structural need for more than 400 ms, write to [arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com) before submitting instead of engineering around the cap.

Verify:

```bash
uv run whest run --estimator estimator.py --residual-wall-time-limit 0.4 --profile
```

## My FLOP counts jumped after updating the kit (flopscope 0.9 / whestbench 0.13)

Symptom: the same estimator reports ~2× `flops_used` after `uv sync`, or a
run that used to fit now hits `budget_exhausted`.

Why it happens: flopscope 0.9 re-priced the cost model (billing is now
`flop_cost × weight × dtype_rate`): float64 bills 2× float32, transcendentals
(`exp`, `log`, `x ** y`) bill 16×, gathers and 3-arg `where` bill 4×/element
(sorts bill ≈4·N·⌈log₂N⌉), and formerly-free ops (`ones`, `stack`,
`concatenate`, copies, `astype`) now bill 1×/element, with `astype` at the
**destination** dtype's rate, so a float32→float64 cast bills 2×/element.
`eye` is billed too, but only for the diagonal cells it writes (`min(m, n)`),
so `fnp.eye(1024)` costs 2,048 FLOPs at its default float64 (1,024 at
float32), not a million. Your estimator does not perform more arithmetic than
before; flopscope 0.9 prices that arithmetic differently.
Scores produced under whestbench 0.12.x are not comparable to 0.13.x.

Fix now: keep estimator state in float32 (`dtype=fnp.float32` on `zeros`/
`ones`/RNG draws); replace `x ** 2` with `x * x`; check `ctx.summary()` for
which ops dominate. If you use exotic dtypes and see `UnsupportedDtypeError`,
move to a standard float dtype. A `CostFallbackWarning` means flopscope
over-charged (never under-charged) a structure-priced op it couldn't bound.

Verify:

```bash
uv run whest run --estimator estimator.py --profile
```

The per-namespace breakdown shows the re-priced ops; utilization
(`mean_compute_utilization`) tells you whether the jump matters (below 0.1
you're still at the multiplier floor and your score is unchanged).

## Result array too large

Symptom: on the **grading server** (not local runs), a per-MLP failure whose message reads `result array too large: N bytes exceeds 4294967296 byte limit`, or `array too large: …` for an input you build. It can also fail your submission at the **smoke test**, before grading starts.

Why it happens: the flopscope runtime on the grading sandbox caps any single array at **4 GiB** (a memory-safety guard). It applies to both arrays you build via `flopscope.numpy` and the array you return from `predict()`. A single array that large almost always means an over-vectorized "all layers × all samples at once" buffer, or `float64` where the MLP weights are `float32`.

Local `whest run` uses an in-process backend **without** this cap, so you won't reproduce it locally; keep your peak single-array size under 4 GiB as a rule. The solution process also gets **8 GB of RAM** in total on the grader, so your peak *combined* footprint matters too, not only the largest single array.

Fix now:

- process samples/rows/columns in **blocks** and accumulate running statistics instead of materializing one giant array,
- keep arrays `float32` (a stray `float64` buffer is 2× larger),
- reshapes and allocations cost **0 FLOP**, so chunking is free against your FLOP budget.

## Class not found

Symptom: `No BaseEstimator subclasses found in <path>.` — or `Ambiguous estimator classes in <path>: A, B. Pass class_name to select one explicitly.`

Why it happens: the loader scans your entrypoint file for classes that subclass `BaseEstimator`. The first message means it found none: you forgot `class Estimator(BaseEstimator):`, or the class lives in an imported helper rather than in the entrypoint. The second means it found several and none is named `Estimator`.

Fix now: for the first, make your class subclass `BaseEstimator` **in the entrypoint file**. For the second, name the one you want `Estimator`, or pass `--class MyClass`. A single subclass is picked up under any name; the name only matters when there is more than one.

Verify:

```bash
uv run whest validate --estimator estimator.py
```

## Packaging / submission rejected (`IMPORT_FAILED`)

Symptom: your submission is rejected before any MLP runs, with a message such as `IMPORT_FAILED`, a `manifest.json` schema / `api_version` error, `'estimator.py' sha256 mismatch`, or "tarball missing manifest.json".

Why it happens: the grader unpacks your archive and checks it against the `manifest.json` it expects: entrypoint, declared versions, and a SHA-256 for every file. A hand-rolled or hand-edited archive (wrong layout, a file changed after the manifest was generated, a missing or stale manifest) fails this gate, so nothing is graded.

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

Why it happens: your estimator imports something the grader sandbox doesn't provide. At grading time only `flopscope` (incl. `flopscope.numpy as fnp`), the `whestbench` API (`BaseEstimator`, `MLP`, `SetupContext`), and the Python standard library are importable. There is **no `requirements.txt` install step**, so third-party packages (`numpy`, `scipy`, `torch`, …) are not available. (A missing *helper module* is different: it ships if you package the folder with `--estimator .` instead of the single file.)

Fix now: for all array math use `import flopscope as flops` and `import flopscope.numpy as fnp` (not `import numpy`). Ship multi-file estimators as a folder. For work that genuinely needs a third-party library (a PyTorch-trained model, a scipy routine), compute it **offline** before packaging and ship the result as a pickle-free `.npz`, loaded in `setup()` (see [Ship Weights](../how-to/ship-weights.md)). `whest validate` runs in your local venv, which *has* numpy/scipy/torch, so it will **not** reproduce a grader-only missing-package error; the only fix is to not import those packages.

Verify:

```bash
uv run whest validate --estimator estimator.py
```

## Signature mismatch

Symptom: `TypeError: predict() missing 1 required positional argument`.

Why it happens: your `predict` method has the wrong signature.

Fix now: ensure signature is `def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:`.

Verify:

```bash
uv run whest validate --estimator estimator.py
```

## Predict raised an unexpected exception

Symptom: `whest run` exits with status `1` and prints an "Estimator Errors" panel listing one or more MLPs with a `PREDICT_ERROR` code. A stderr line reads, for example, `2 of 10 MLP(s) raised during predict; rerun with --debug for tracebacks...`.

Why it happens: your `predict()` raised an exception that is neither `BudgetExhaustedError` nor `TimeExhaustedError`. WhestBench routes the failure through the zero-prediction path: the affected MLP scores `final_layer_mse_m × 1.0` (no compute discount) and the suite mean stays finite. The non-zero exit code signals that the submission is not yet passing.

Fix now:

```bash
# Show full tracebacks in the "Estimator Errors" panel:
uv run whest run --estimator estimator.py --debug

# Stop at the first failure and propagate the raw Python traceback:
uv run whest run --estimator estimator.py --debug --fail-fast
```

The traceback in the panel (or the raw stack from `--fail-fast`) points directly at the line in your estimator that raised.

## Setup failed (`SETUP_ERROR` / `SETUP_FAILED`)

Symptom: locally, `whest run` prints `Error [setup:SETUP_ERROR]: <your exception's message>`; on the grader it is reported as `SETUP_FAILED: <Exception>`. Either way, the submission is rejected before any MLP is scored.

Why it happens: your estimator's `setup()` raised. Unlike a `predict()` failure (which is isolated to a single MLP and zero-scored), an exception in `setup()` rejects the **whole** submission: there is nothing to grade if setup never completes. Common causes: a weights/`.npz` file that didn't ship or loads with the wrong path, an assertion or config read that's brittle on the grader, or work that's fine locally but trips on a sandbox difference.

Fix now: make `setup()` exception-proof. Load files via the path the framework gives you, guard fallible work defensively (try/except with a safe fallback), and keep it to loading rather than computing; heavy precompute belongs **offline**, shipped as an artifact you read here (it also avoids the [setup timeout](#setup-timeout)). Don't move it into `predict()` instead: `predict()` is the billed side. Reproduce locally with the isolated runner before submitting:

```bash
uv run whest run --estimator estimator.py --runner subprocess --debug
```

## Setup timeout

Symptom: `SETUP_TIMEOUT` error.

Why it happens: on the grader, `setup()` (plus your module import) has a hard **5 second** budget, and it overran. This is session-level: it fails the **whole** submission, not one MLP.

`setup()` is not a once-per-submission hook. It runs once per worker process, roughly **5-15 times** per submission, since a submission is served by several workers and a worker that dies mid-suite is replaced and re-runs `setup()`. Each run gets its own 5 s, and any one of them overrunning fails the submission. Keep it idempotent and cheap on every call, not only the first. See [Estimator Contract: Lifecycle](../reference/estimator-contract.md#lifecycle).

Locally, `whest run` and `whest validate` use the same 5 s default, so a local `SETUP_TIMEOUT` means what a graded one does. To see your headroom, rehearse a tighter cap with `--setup-timeout`. A setup that lands close to the ceiling is already a problem: the grader's machine is not guaranteed to be as fast as yours.

Fix now: precompute offline and ship the result as an artifact your `setup()` loads (see [Ship Weights](../how-to/ship-weights.md)). Do **not** move the work into `predict()`: `setup()` is off the FLOP budget and off the residual, `predict()` is billed for both, so that trade makes your score worse.

Verify:

```bash
uv run whest run --estimator estimator.py --runner local --debug
```

## Predict timeout

Symptom: `PREDICT_TIMEOUT` or `TIME_EXHAUSTED` error.

Why it happens: a single `predict()` call exceeded its wall-clock limit. **On the grader each MLP gets 120 seconds.** Overrunning it fails that MLP: its predictions are zeroed and its multiplier is forced to 1.0, exactly like exceeding the FLOP budget.

This is a per-call safety guardrail measured in wall-clock seconds, not the FLOP budget; the two are independent, and you can fail either one on its own.

The local runners default to the grader's 120 s cap. Confirm what you ran under in `run_config.wall_time_limit_s` in the JSON report, and pass `--wall-time-limit` only when you want a tighter local cap.

There is one exception on **whestbench 0.14.0 or older**: those versions give up after 30 seconds under `--runner subprocess`, well short of the real limit, so a `predict()` taking longer than 30 seconds fails locally even though it would pass on the grader. This kit now pins whestbench 0.16.0, where the subprocess runner waits for the full wall limit plus a grace margin, so a local subprocess `PREDICT_TIMEOUT` again means what a graded one does. If you are on an older kit, run `uv sync` before trusting a timeout near 30 seconds.

Fix now: check for infinite loops or extremely expensive operations. If you are close to the limit, treat that as a warning: the cap is per MLP, and a call that fits on one run can overrun on the next.

Verify:

```bash
uv run whest run --estimator estimator.py --runner local --debug
```

## Whole-submission timeout (`WATCHDOG_TIMEOUT`)

Symptom: `WATCHDOG_TIMEOUT` on some or all of your per-MLP entries, on a submission that was still making progress.

Why it happens: separately from the per-MLP limits above, the grader bounds the evaluation as a whole. It finalizes your submission when **either**:

- **no MLP has finished for 8 minutes** (a no-progress timer, reset every time an MLP completes), or
- **the evaluation has been running for 40 minutes in total** (a hard cap, regardless of progress).

Whichever fires first, every MLP not yet finished is recorded as `WATCHDOG_TIMEOUT`. Those count as failed MLPs, so their predictions are zeroed and their multiplier is forced to **1.0**, the same outcome as any other per-MLP failure. MLPs already scored keep their scores.

These are the aggregate limits. In particular there is no per-worker time budget: the 40 minutes is one wall-clock ceiling on the whole submission, measured from when your evaluation starts, and it is not divided among the machines that run it. Both are grader-side operational limits, not whestbench settings; they can change without a kit release.

One further submission-level bound is not a watchdog at all: if the cluster never serves your session, it is finalized with `truncation.reason = "no_worker_capacity"` rather than `no_progress`. That one is a grader-side fault: nothing in your submission caused it, and there is no partial result to read. Re-submit.

You are unlikely to reach either limit with a working estimator: the 120 s per-MLP cap bounds each call and MLPs are graded in parallel. But an estimator that sits near the per-MLP cap on every MLP has no margin left. These limits exist to terminate a genuinely stuck evaluation rather than to constrain a slow one. If you do see `WATCHDOG_TIMEOUT`, the usual causes are an estimator that hangs (see **Predict timeout** above) or a transient grader-side infrastructure fault.

Fix now: if only some MLPs carry the code and the rest scored normally, re-submit; a partial `WATCHDOG_TIMEOUT` is more often infrastructure than your code. If every MLP carries it, look for a hang in `predict()` or `setup()`.

Verify: check `n_failed_mlps` and the per-MLP `error_code` values in the score report.

(Local `whest run` has no equivalent timer, so this only appears on the server.)

## Budget exhausted mid-operation

Symptom: `BudgetExhaustedError` raised during a specific operation.

Why it happens: a single flopscope operation would exceed your remaining FLOP budget.

Fix now: use `flops.budget_summary()` to find the expensive operation. Consider diagonal approximations or fewer iterations.

Verify: check `flops_used` in the score report.

## Numerical instability in deep networks

Symptom: predictions become `nan` or `inf` after many layers.

Why it happens: values grow or shrink exponentially through deep networks without safeguards.

Fix now: add overflow guards, rescaling the covariance when diagonal values exceed a threshold (see [examples/03_covariance_propagation.py](../../examples/03_covariance_propagation.py)). Rescaling, not widening, is the cheap fix: float64 does buy exponent range, but it bills exactly 2× per element on every metered op, and the 03 example reaches the same final-layer MSE to five significant figures at half the FLOPs by staying in float32. If one specific reduction genuinely needs the range, widen only that operation and cast back with `.astype(fnp.float32)`.

Verify:

```bash
uv run whest validate --estimator estimator.py
```

## Dtype mismatch

Symptom: output is float64 but evaluator expects float32, or similar type issues.

Why it happens: flopscope operations may produce different dtypes than expected.

Fix now: cast your output: `return fnp.asarray(result, dtype=fnp.float32)`.

Verify:

```bash
uv run whest validate --estimator estimator.py
```

## Empty predictions

Symptom: returned shape `(0, width)` or similar zero-length array.

Why it happens: your layer loop did not iterate (empty `mlp.weights`).

Fix now: check that you iterate over `mlp.weights` and append results per layer.

Verify:

```bash
uv run whest validate --estimator estimator.py
```

## Using numpy instead of flopscope

Symptom: operations work but FLOP budget is not consumed (shows 0 flops_used).

Why it happens: you are using `import numpy as np` instead of `import flopscope.numpy as fnp`. Numpy operations are not FLOP-tracked. (This is the *local* symptom: your venv has numpy, so it runs silently untracked. On the grader `import numpy` fails outright, since the sandbox has no numpy; see [Import error in estimator](#import-error-in-estimator). Either way, `np.*` is wrong; use `fnp.*`.) Shipping numpy, or any other array library or compiled kernel, alongside your estimator to make `np.*` work is **prohibited** and disqualifiable: the requirement is that your math is metered, not merely that it runs.

Fix now: replace all `np.*` calls with `fnp.*` equivalents. See [Code Patterns](../reference/code-patterns.md).

Verify: check `flops_used > 0` in score report.

## No numpy in the sandbox (`No module named 'numpy'` / `name 'np' is not defined`)

Symptom: on the grader, `ModuleNotFoundError: No module named 'numpy'` — or, if your `import numpy as np` was wrapped in a `try`/`except` that swallowed it, a later `NameError: name 'np' is not defined` at the first `np.*` call.

Why it happens: the grading sandbox does **not** ship raw numpy. It provides `flopscope.numpy` (a FLOP-counting, numpy-compatible proxy) instead. `import numpy` has nothing to import, so it raises. A swallowed import leaves `np` undefined, which surfaces as a `NameError` further down. (See also the broader [Import error in estimator](#import-error-in-estimator) for the full list of what the sandbox does and doesn't provide.)

Fix now: import the flopscope array module and write all array code against it:

```python
import flopscope.numpy as fnp     # or: import flopscope.numpy as np
```

Bind it to `np` if you don't want to rename every call site, but do not `import numpy` and do not silently `except ImportError` around it.

Verify:

```bash
uv run whest run --estimator estimator.py --runner subprocess
```

## flopscope arrays are immutable

Symptom: `flopscope arrays are immutable, so item assignment ... is not supported`, or `in-place add (arr += x) is not supported` (and similar for other in-place operators).

Why it happens: arrays produced by `flopscope.numpy` are read-only proxies. Mutating ops (`arr[i] = v`, `arr[mask] = v`, `arr += x`, `arr *= x`) are rejected by design, on the grader and (with the client) locally.

Fix now: build new arrays functionally instead of mutating in place:

- replace `arr[i] = v` / `arr[mask] = v` with `fnp.where(mask, v, arr)`, or rebuild via slicing + `fnp.concatenate`,
- replace `arr += x` with `arr = arr + x` (and likewise `*=`, `-=`, `/=`),
- set a diagonal with `fnp.fill_diagonal(M, v)`, the one sanctioned in-place write, and cheap (`min(m, n)` FLOP). Like NumPy's, it returns `None` and edits `M` in place, so call it as a statement: `fnp.fill_diagonal(M, v)`, **not** `M = fnp.fill_diagonal(M, v)`.

See [Code Patterns](../reference/code-patterns.md) for the functional equivalents.

Verify:

```bash
uv run whest run --estimator estimator.py --runner subprocess
```

## Reduction `axis` must be an int or tuple, not a list

Symptom: `TypeError: 'list' object cannot be interpreted as an integer` from a reduction such as `sum`, `mean`, `max`, or `prod`.

Why it happens: numpy itself only accepts `axis=<int>`, `axis=<tuple>`, or `axis=None` for reductions; a `list` axis is rejected. flopscope mirrors numpy exactly, so passing a list fails the same way it would in plain numpy.

Fix now: pass a tuple, not a list:

```python
a.sum(axis=(0, 1))     # correct
a.sum(axis=[0, 1])     # TypeError
```

Verify:

```bash
uv run whest run --estimator estimator.py --runner local --debug
```

## Every MLP failed (n_failed_mlps == n_mlps)

Symptom: the suite-level `failure_breakdown` shows every MLP carrying at least one failure flag (`n_failed_mlps == n_mlps`), and the `adjusted_final_layer_score` is dominated by `MSE(0, Y_m) × 1.0` across the board (typically lands near `0.91`, the raw `final_layer_mse` of zero predictions at the default activation scale).

> **Note:** this is the post-merge replacement for the older "score is `inf`" symptom. Since whestbench PR #39 (May 2026) failures produce finite scores at the zero-prediction × 1.0 multiplier; there is no longer an `inf` path.

Why it happens: every MLP either raised during `predict()` or exhausted the FLOP / wall-time / residual-wall-time / combined budget.

Tell them apart from `failure_breakdown` and the exit code:

- **Exit `1` + non-zero `failure_breakdown.error` + "Estimator Errors" panel** — `predict()` raised exceptions on at least one MLP. See [Predict raised an unexpected exception](#predict-raised-an-unexpected-exception).
- **Exit `0` + every `per_mlp[i].budget_exhausted: true`** — you ran out of analytical FLOPs.
- **Exit `0` + every `per_mlp[i].combined_budget_exhausted: true`** — the post-hoc check on effective compute (`C_m = F_m` in Phase 2) exceeded the cap even though flopscope itself didn't trip.
- **Exit `0` + every `per_mlp[i].residual_wall_time_exhausted: true`** — you exceeded the 400 ms residual wall-time cap.
- **Exit `0` + every `per_mlp[i].time_exhausted: true`** — you ran out of wall-clock time.

Fix now: run with `--debug` to see tracebacks in the "Estimator Errors" panel (works with any runner), or `--fail-fast` to halt at the first failing MLP with the raw Python stack:

```bash
uv run whest run --estimator estimator.py --debug
uv run whest run --estimator estimator.py --debug --fail-fast
```

## Setup runs expensive operations

Symptom: a slow `setup()`, or a `SETUP_TIMEOUT` you didn't expect.

Why it happens: on the grader, `setup()` runs in its own flopscope session, which is opened and closed around it before any per-MLP budget exists. Its FLOPs are measured but billed to no MLP, and its wall time is not charged to `residual_wall_time_s`. So expensive work in `setup()` will **not** show up as budget consumption — that is not the failure mode to look for.

What it does cost is time: a hard 5 s ceiling on every `setup()` run, and overrunning it fails the whole submission. `setup()` runs once per worker process (roughly 5-15 times per submission), so expensive work there is paid for again on each one.

Fix now: keep `setup()` to loading rather than computing. Precompute offline and ship the artifact (see [Ship Weights](../how-to/ship-weights.md)). Moving the work into `predict()` is the wrong direction: that side is billed for both FLOPs and residual.

## ➡️ Next step

- [Debugging Checklist](../how-to/debugging-checklist.md)
- [FAQ](./faq.md)
- [Estimator Contract](../reference/estimator-contract.md)
- [Scoring Model](../concepts/scoring-model.md)

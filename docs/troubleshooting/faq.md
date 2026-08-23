# Frequently asked questions

> [← Documentation](../README.md)

## Can I use numpy directly?

No. Plain `import numpy` is **not available** in the grader sandbox (by design). All array math goes through flopscope (`import flopscope as flops` and `import flopscope.numpy as fnp`), which wraps numpy with analytical FLOP counting. Your score depends on the FLOP cost of your operations, and only flopscope tracks those costs, so flopscope is both the only array path and the only one that counts.

Shipping your own copy is not a way around it. Vendored array libraries, compiled kernels of any kind, `ctypes`/`cffi`/FFI, and anything else that computes outside flopscope are **prohibited**: a submission that does it is disqualified, not merely charged for the time it took. Shipped **data** (weights, lookup tables, precomputed artifacts) remains entirely welcome; it is shipped *code that computes* that is not. The full boundary is in [Allowed Code](../concepts/allowed-code.md).

## Can I use scipy (or PyTorch, or any other PyPI package)?

No. At grading time your estimator runs in a locked-down sandbox whose only importable libraries are `flopscope` (incl. `flopscope.numpy as fnp`), the `whestbench` API (`BaseEstimator`, `MLP`, `SetupContext`), and the Python standard library. There is **no `requirements.txt` install step**: third-party packages (`scipy`, `numpy`, `torch`, …) are not installed and won't import. For the standard normal CDF, use the pure-flopscope `norm_cdf` recipe in [Code Patterns](../reference/code-patterns.md#standard-normal-cdf). For anything heavier, such as a model trained with PyTorch, do the work **offline** before packaging and ship the result as a pickle-free `.npz`, loaded in `setup()` via `fnp.load(str(path))` (0 FLOPs); see [Ship Weights](../how-to/ship-weights.md). Bundling the package into your submission instead is prohibited and disqualifiable: ship the *result*, never the library.

## Why is one MLP scoring much worse than the others?

A per-MLP `adjusted_final_layer_score` that is much higher than the others almost always means that MLP **failed**: your estimator raised, exceeded the FLOP budget, exceeded the wall-time cap, returned the wrong shape, or returned non-finite values. WhestBench treats every failure as if your estimator had returned a zero array and forces the per-MLP multiplier to **1.0** (no compute discount). Concretely: `adjusted_final_layer_score_m = MSE(0, Y_m) × 1.0` for the failed MLP, which is strictly worse than a trivial-zero submission that succeeds (which gets the 0.1 multiplier floor).

The suite mean stays finite: one failed MLP no longer poisons the whole run, but it does pull the mean noticeably toward the raw `final_layer_mse` of the zero prediction (`0.9095`, measured across the `@v2-phase2` Mini split).

Diagnose by reading the failure flags on the failing per-MLP entry: `budget_exhausted`, `time_exhausted`, `residual_wall_time_exhausted`, `combined_budget_exhausted`, `error` / `error_code` / `traceback`. The suite-level `failure_breakdown` gives counts per flag, and `n_failed_mlps` is the total count of MLPs that hit any failure path.

To see tracebacks, run with `--debug`; to halt at the first failure, add `--fail-fast`:

```bash
uv run whest run --estimator estimator.py --debug
uv run whest run --estimator estimator.py --debug --fail-fast   # halt at first error
```

See [Estimator Contract: Failure semantics](../reference/estimator-contract.md#failure-semantics) for the complete list of failure paths.

## Do I need to use the `budget` argument in `predict()`?

No. flopscope enforces the budget whether you read the argument or not: if your
operations exceed it, `BudgetExhaustedError` is raised and your predictions are
zeroed.

The `budget` argument tells you how many FLOPs you are allowed. It's usually best
to use it as a fixed hard cap and stay with one strategy throughout the run.

## Can I precompute things in `setup()`?

Yes, and the FLOPs really are free: `setup()` runs outside the per-MLP FLOP budget, and its wall time is not charged to `residual_wall_time_s` either.

The constraint is wall clock. `setup()` runs under a hard **5 s** ceiling, and overrunning it fails the whole submission with `SETUP_TIMEOUT` — not one MLP, all of them.

And it is not a once-per-submission hook. The grader runs `setup()` once per worker process, a submission is served by several workers, and a worker that dies mid-suite is replaced and re-runs it — roughly **5-15 times** per submission in practice. Each run gets its own 5 s, so anything expensive is paid for again on every worker. Keep it idempotent and cheap on every call. See [Estimator Contract: Lifecycle](../reference/estimator-contract.md#lifecycle).

That makes `setup()` the right place to load MLP-independent work (weights, lookup tables, configuration) and the wrong place to generate it. If the precompute is expensive, do it offline and ship the result (see [Ship Weights](../how-to/ship-weights.md)).

## I added a helper module or weights file, but it didn't end up in my submission.

`whest package --estimator estimator.py` (and `whest submit --estimator estimator.py`) ship **only that one file**. To ship more than one file, keep them in a folder and point `--estimator` at the **folder**:

```bash
uv run whest package --estimator . --output submission.tar.gz
```

You'll see the full list of files before anything is sent, and credential files like `.env` are never included.

Since whestbench 0.16.0 this cannot happen silently for a file you import: `whest package` prints `⚠ Single-file submission: only estimator.py will be submitted. N items beside it will NOT be included: …`, naming each one, and it **refuses** with `Error [package:PACKAGING_VALIDATION_ERROR]` if your single file imports one of them. The fix is the same: point `--estimator` at the folder. The warning stays advisory for files your estimator does not import (a weights `.npz` opened by a computed path, say), so still read the list.

See [Ship Weights and Multi-File Submissions](../how-to/ship-weights.md).

## How do I set a time limit on my estimator code?

At the flopscope level, time limits live on `BudgetContext` via
`wall_time_limit_s`:

```python
import flopscope as flops

with flops.BudgetContext(flop_budget=10_000_000, wall_time_limit_s=2.0) as budget:
    ...
```

In WhestBench CLI runs, `--wall-time-limit` sets that same limit for each
`predict()` call.

## What is `residual_wall_time_limit`?

`residual_wall_time_limit` is a WhestBench rule, not a `BudgetContext`
parameter. flopscope reports:

- `flopscope_backend_time_s`: time spent inside counted flopscope calls
- `flopscope_overhead_time_s`: time spent inside flopscope's own dispatch
- `residual_wall_time_s`: participant Python (loops, control flow), GC, and Python-callback op time; as of flopscope 0.7.0, data-movement NumPy ops (concatenate, stack, tile, repeat, take, pad, …) count as `flopscope_backend_time_s`, not residual

In Phase 2 that limit is a **hard cap of 400 ms per MLP**, and it is a cap
rather than a price: residual wall time is not converted into FLOPs and charged
to your budget. Overrun it and WhestBench zeros your predictions for that MLP
and forces the multiplier to **1.0** (`residual_wall_time_exhausted: true`),
the same outcome as any other per-MLP failure.

The cap is sized for **plumbing**: control flow, slicing inputs into blocks,
bookkeeping between flopscope calls. It is not an allowance for computation.
Doing meaningful compute in residual time (raw Python arithmetic over neurons,
a vendored library, a compiled kernel, a subprocess) is prohibited and
disqualifiable, whether or not it happens to fit inside 400 ms (see
[Allowed Code](../concepts/allowed-code.md)).

Rehearse the cap locally by passing it explicitly:

```bash
uv run whest run --estimator estimator.py --residual-wall-time-limit 0.4
```

If your estimator has a legitimate structural need for more residual time than
the cap allows, don't engineer around it; write to
[arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com) and describe the case before you submit.

## What happens if I exceed the FLOP budget?

flopscope raises `BudgetExhaustedError` before the over-budget operation executes. The framework catches this, zeros all your predictions for that MLP, and forces the per-MLP multiplier to **1.0** (no compute discount). You will see `budget_exhausted: true` in the per-MLP report and `adjusted_final_layer_score_m = final_layer_mse_m × 1.0` for the affected MLP. There is also a **post-hoc** combined-budget check: after `predict()` returns, the scoring layer re-checks effective compute against the budget and surfaces `combined_budget_exhausted: true` (same zero/×1.0 outcome). In Phase 2 effective cost is `C_m = F_m` (residual wall time is capped at 400 ms, not priced into it), so that check is a backstop on the same FLOP count; blowing the residual cap surfaces separately as `residual_wall_time_exhausted`.

## Is there a memory limit?

Yes, two. Your solution process gets **8 GB of RAM** in total, and the grading sandbox additionally caps any single array at **4 GiB**, on both the arrays you build via `flopscope.numpy` and the array you return from `predict()` (at smoke and grading). The array cap is a memory-safety guard, not a scoring rule, and it sits far above what an efficient estimator needs. If you hit `result array too large`, chunk into row/column blocks; reshapes and allocations cost 0 FLOP, so it's free against your budget. (Local `whest run` has no such cap, so this only appears on the server.)

## Is there a limit on total evaluation time?

Yes, two that can cut a running evaluation short, and they apply to the submission as a whole rather than to any one MLP. The grader finalizes your evaluation when **no MLP has finished for 8 minutes**, or when the evaluation has been running for **40 minutes in total**, whichever comes first. Any MLP not finished by then is recorded as `WATCHDOG_TIMEOUT` and counts as a failed MLP (predictions zeroed, multiplier forced to 1.0); MLPs already scored keep their scores. These are grader-side operational limits, not whestbench settings; they can change without a kit release.

A third submission-level bound is not a watchdog at all: if the cluster never serves your session, it is finalized with `truncation.reason = "no_worker_capacity"` rather than `no_progress`. That one is a grader-side fault: nothing in your submission caused it, and there is no partial result to read. Re-submit.

There is **no per-worker time budget**. The 40 minutes is a single wall-clock ceiling on the whole submission, not a per-machine allowance that gets divided up. A working estimator does not come close: the 120 s per-MLP cap bounds each call and MLPs are graded in parallel. But an estimator that sits near the per-MLP cap on every MLP has no margin left. The limits are there to terminate a stuck evaluation, not to constrain a slow one. See [Whole-submission timeout](common-participant-errors.md#whole-submission-timeout-watchdog_timeout). (Local `whest run` has no equivalent timer, so this only appears on the server.)

## How do I inspect budget summaries while debugging?

Use:

- `budget.summary()` for the current explicit `BudgetContext`
- `flops.budget_summary()` for the accumulated process/session view
- `budget.summary_dict(...)` or `flops.budget_summary_dict(...)` for structured data

If you want namespace attribution, pass `by_namespace=True`.

## Is scoring hardware-dependent?

Partly.

**`F_m`, the analytical FLOP count, is not.** flopscope derives it from tensor
shapes and dtypes, so it is identical on your laptop and on the grader.

**`R_m`, the residual wall time, is.** Phase 2 does not price it (effective
compute is `C_m = F_m`), but it does cap it, at **400 ms per MLP**, and
overrunning the cap zeroes that MLP. Everything flopscope does *not* meter
(pure-Python loops, control flow, GC, I/O) runs at the speed of whatever machine
the grader hands you, so a loop sitting at 300 ms on your laptop can trip a cap
you thought you were under. Moving real computation into that window is not an
option either: raw `numpy`, compiled extensions you ship yourself, threads,
subprocesses, and multiprocessing are **prohibited** in Phase 2, and using them
is disqualifiable rather than merely expensive.

The rules do not guarantee any particular evaluation hardware. Do not assume
the grader matches your development machine in core count, clock speed, or
BLAS backend. A submission whose cost or behavior depends on a fixed amount
of wall clock is carrying a risk it does not need to carry.

The way to stay hardware-independent is to keep the work inside
`flopscope.numpy`, where it is metered analytically and your local number is
the grader's number. Anything you move outside it is timed instead of metered.

**If your estimator has an internal deadline** (`time.time()` checks, a
`PREDICT_TIMEOUT` constant, an anytime algorithm that stops refining when the
clock runs out), do not calibrate it against your own machine's speed. On
slower hardware a locally-tuned deadline trips early and your estimator
silently returns its fallback answer: no exception, no failed MLP, only a
valid-looking score computed from the degraded path. Prefer a **budget-based**
cutoff over a clock-based one: `predict()` is handed `budget`, and
`flops.budget_summary()` tells you what you have spent. If you must use a clock, make the
fallback something you would be content to be scored on.

To see where you stand, run with `--profile` and read `residual_wall_time_s`
against the 400 ms cap. A conservative local check is `--max-threads 1`, which
takes away the BLAS parallelism you cannot count on having.

## How many MLP networks are in a full evaluation?

Two different numbers, so be careful which one you are sizing against:

- **Locally**, `whest run` defaults to **10** MLPs (configured by `n_mlps` in `ContestSpec`). Raise it with `--n-mlps` if you want a closer rehearsal.
- **On the grader**, a full evaluation scores **100** MLPs, split into a public half that drives the visible leaderboard and a held-out half.

Each MLP has the same width and depth but different random weights and a distinct grader-supplied `mlp.seed` for any estimator-side randomness. Your aggregate score is the mean of the per-MLP `adjusted_final_layer_score` values.

The FLOP budget is **per MLP**, not shared across the run, so scoring 100 instead of 10 does not shrink the budget available to any one of them. What it does change is total time: see [Is there a limit on total evaluation time?](#is-there-a-limit-on-total-evaluation-time).

## What if my estimator is fast but inaccurate?

You are ranked by the **budget-adjusted** `adjusted_final_layer_score = final_layer_mse × max(0.1, C_m / B)`, not raw MSE. Using less than 10% of the effective-compute budget gets you the 0.1 multiplier floor, a factor-of-ten discount and no more. So extremely cheap and inaccurate beats moderately cheap and inaccurate only up to that floor; below it, there is no further benefit to being cheaper.

## My local score is great but my submission scores 10x worse — why?

Almost always one of four things:

1. **State carries between MLPs locally, but the grader spreads your suite over several workers.** Module-level and instance state survives from one `predict()` to the next in *both* local runners: one worker serves the whole local suite, `setup()` runs once, and `--runner subprocess` behaves exactly like `--runner local` here. The grader does not. It serves one submission from several worker processes, so `setup()` runs roughly 5-15 times and each worker sees only the MLPs it was handed; a worker that dies mid-suite is replaced and starts over. A cache that assumes it has seen every previous MLP is therefore populated locally and empty in production. **Fix:** make each `predict()` correct on its own, and treat anything built across calls as an optimization that may be missing. Populate what you can in `setup()` from `self._...` attributes, and ship precomputed data as a file loaded from `submission_dir` ([Ship Weights](../how-to/ship-weights.md)). Do **not** plan caching around `SetupContext.scratch_dir`: it is reserved and `None` on every whestbench 0.16.0 path, so touching it raises inside `setup()`, which fails the entire submission rather than one MLP.

2. **Imports that work locally fail in the grader sandbox.** Two flavors: (a) a helper module that didn't ship (you packaged the single file instead of the folder) or a side-effecting top-level statement, caught by running `uv run whest run --estimator estimator.py --runner subprocess` locally before submitting, then reading the "Estimator Errors" panel; (b) an `import` of a package the grader doesn't provide. The sandbox has **only** `flopscope`, the `whestbench` API, and the Python stdlib: no `numpy`/`scipy`/`torch`. Your local venv *does* have those, so a local run won't flag them; the fix is to not import them (use `flopscope.numpy as fnp`, and precompute heavier work offline; see [Ship Weights](../how-to/ship-weights.md)).

3. **Numerical non-determinism without a seed.** Random MLP generation, Monte-Carlo ground truth, or your estimator's own RNG. **Fix:** add `--seed N` to your local runs so successive runs are comparable, and avoid time-based seeds in your estimator.

4. **Work that flopscope never metered.** Raw `numpy`, a bundled BLAS, threads, and `multiprocessing` are **prohibited** in Phase 2: a submission that computes outside flopscope is disqualified, not re-priced. Even legitimate plumbing (Python loops, control flow, GC, I/O) runs at whatever speed the grader's machine gives it, and `residual_wall_time_s` can be several times larger there than at home. Symptom: `flops_used` matches your local run exactly, while `residual_wall_time_exhausted` fires on MLPs that passed locally. A clock-based internal deadline can also trip early and silently substitute a fallback answer. **Fix:** move the math into `flopscope.numpy` so it is metered analytically, and rehearse with `--residual-wall-time-limit 0.4 --max-threads 1`. See [Is scoring hardware-dependent?](#is-scoring-hardware-dependent).

If your Stage 3 and Stage 4 scores agree but the grader still disagrees, suspect Python-version or BLAS-version drift; `uv run whest doctor` will surface the relevant runtime info.

## How many submissions can I make per day?

**Ten per team per UTC day.** The counter resets at 00:00 UTC, and a submission
that fails on the grader still spends one, so rehearse at
[Stage 4](../getting-started/stage-4-run-subprocess.md) before you spend a slot.

## Who do I contact about a rules question or a FLOP that looks mispriced?

Email [arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com). Use it for:

- **rules questions** — whether the thing you want to build is allowed, *before*
  you build it;
- **suspected FLOP mispricing** — an operation whose billed cost doesn't match
  what it should be. Send a minimal repro: the op, the shapes and dtypes, and
  the `ctx.summary()` rows you're reading it from;
- **a legitimate need for more residual wall time than the 400 ms cap allows** —
  describe the structural reason, before you submit rather than after.

Anything that can be discussed in the open — scoring questions, "why is my MSE
like this", packaging trouble — moves faster on the
[challenge discussion forum](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026).

## Did the scoring formula change when the kit updated?

No. `adjusted_final_layer_score = final_layer_mse × max(0.1, C_m / B)` is
unchanged. What changes between kit releases is the *meter*, and it has moved
several times. whestbench 0.13.0 adopted flopscope 0.9 (float64 2×,
transcendentals 16×, formerly-free data movement now billed). whestbench 0.14.0
adopted flopscope 0.10: `out=` destinations priced, symmetry-tag discounts voided
when the tagged buffer is written (re-validate with `as_symmetric()` to keep them,
as `examples/03_covariance_propagation.py` shows), and `einsum(..., out=)` following
NumPy's casting rules so calls that used to silently truncate now raise
`TypeError`. whestbench 0.15.0 adopted flopscope 0.11: a contraction past the
52-letter subscript budget now bills the full fused-multiply-add count instead of
multiplies only, `ix_` moved from free to billed, and non-numeric dtypes
(`object`, `str_`, `bytes_`, `datetime64`, structured) now raise
`UnsupportedDtypeError` anywhere they reach a metered op. Convert those with plain
`numpy` before handing data to flopscope.

Most recently, 0.16.0 adopted flopscope 0.12, with roughly forty operations repriced
in both directions. Costing more: `nan*` reductions now pay for the NaN-scan pass
they always ran (integer and boolean input exempt, and `nanmax`/`nanmin` exempt),
six boolean-answering ops that compute in a promoted float dtype (`signbit`,
`isneginf`, `isposinf`, `isclose`, `allclose`, and `is_symmetric`/`as_symmetric`)
stop undercharging by 2×, a narrow `out=` can no longer lower a unary op's price,
and `angle` on boolean input doubles. Costing less: `where`/`insert` mixed with a
Python scalar follow NEP 50 weak promotion, a refused `percentile`/`quantile`
costs nothing, `ldexp` is priced on its mantissa loop, and a dtype combination
NumPy cannot run is no longer billed before it raises. flopscope ships 0.12.0 as
a new version opening a new phase, so nothing already scored was repriced.

**This kit pins the meter it is built against, which is not always the newest release.**
The kit resolves **flopscope 0.12.0** with **whestbench 0.16.0**: whestbench 0.16.0
declares `flopscope>=0.12.0,<0.13.0`, so the two are an atomic pair rather than
independent choices. The three repos move as one unit, in this order: whestbench cuts
a release naming a flopscope minor, the evaluator re-pins to that pair, and the kit
follows. While a move is in progress the kit and the grader can sit one minor apart, and
FLOP counts you measure locally may not be the ones your submission is scored under.
`pyproject.toml` records which step the kit is on; read the comment at the top of it
before trusting a local total to match the leaderboard.

Whenever the meter moves, `F_m` (and therefore `C_m` and the multiplier) can
change for identical code. Leaderboard scores are re-evaluated under the new
meter; local scores taken under an older kit are not comparable. Re-run
`uv sync` and re-measure rather than trusting a number you recorded earlier. If
the number you are comparing against predates Phase 2, the shape and the cost
model moved too. [Competition Rounds](../reference/rounds.md) lays every round
side by side and gives the flags that reproduce an earlier one.

## ➡️ Next step

- [Common Participant Errors](./common-participant-errors.md)
- [Debugging Checklist](../how-to/debugging-checklist.md)
- [Scoring Model](../concepts/scoring-model.md)

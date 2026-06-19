# Frequently Asked Questions

> [← Documentation](../README.md)

## Can I use numpy directly?

All computation must go through flopscope (`import flopscope as flops` and `import flopscope.numpy as fnp`). flopscope wraps numpy with analytical FLOP counting — your score depends on the FLOP cost of your operations, and only flopscope tracks those costs.

## Can I use scipy?

Yes. scipy is not part of flopscope, so you import it separately as your own dependency. Common usage: `scipy.special.ndtr` for the standard normal CDF. Add `scipy` to your `requirements.txt` when packaging.

## Why is one MLP scoring much worse than the others?

A per-MLP `adjusted_final_layer_score` that is much higher than the others almost always means that MLP **failed** — your estimator raised, exceeded the FLOP budget, exceeded the wall-time cap, returned the wrong shape, or returned non-finite values. WhestBench treats every failure as if your estimator had returned a zero array and forces the per-MLP multiplier to **1.0** (no compute discount). Concretely: `adjusted_final_layer_score_m = MSE(0, Y_m) × 1.0` for the failed MLP, which is strictly worse than a trivial-zero submission that succeeds (which gets the 0.1 multiplier floor).

The suite mean stays finite — one failed MLP no longer poisons the whole run, but it does pull the mean noticeably toward the raw `final_layer_mse` of the zero prediction (`~0.91` at the default network shape).

Diagnose by reading the failure flags on the failing per-MLP entry: `budget_exhausted`, `time_exhausted`, `residual_wall_time_exhausted`, `combined_budget_exhausted`, `error` / `error_code` / `traceback`. The suite-level `failure_breakdown` gives counts per flag, and `n_failed_mlps` is the total count of MLPs that hit any failure path.

Run with `--debug` to see tracebacks; `--fail-fast` to halt at the first failure:

```bash
whest run --estimator estimator.py --debug
whest run --estimator estimator.py --debug --fail-fast   # halt at first error
```

See [Estimator Contract: Failure semantics](../reference/estimator-contract.md#failure-semantics) for the complete list of failure paths.

## Do I need to use the `budget` argument in `predict()`?

The `budget` argument tells you how many FLOPs you are allowed. It's usually best
to use it as a fixed hard cap and stay with one strategy throughout the run.

flopscope enforces the budget regardless — if your operations exceed it, `BudgetExhaustedError` is raised and your predictions are zeroed.

## Can I precompute things in `setup()`?

Yes. `setup()` runs before any `predict()` calls and is not under a FLOP budget. Use it for one-time preparation that does not depend on the specific MLP (e.g., lookup tables, configuration).

## I added a helper module or weights file, but it didn't end up in my submission.

`whest package --estimator estimator.py` (and `whest submit --estimator estimator.py`) ship **only that one file**. To ship more than one file, keep them in a folder and point `--estimator` at the **folder**:

```bash
uv run whest package --estimator . --output submission.tar.gz
```

You'll see the full list of files before anything is sent, and credential files like `.env` are never included. See [Ship Weights and Multi-File Submissions](../how-to/ship-weights.md).

However, `setup()` does have a time limit (`setup_timeout_s`, typically 5 seconds).

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

WhestBench can then zero predictions if `residual_wall_time_s` exceeds the
configured `--residual-wall-time-limit`.

## What happens if I exceed the FLOP budget?

flopscope raises `BudgetExhaustedError` before the over-budget operation executes. The framework catches this, zeros all your predictions for that MLP, and forces the per-MLP multiplier to **1.0** (no compute discount). You will see `budget_exhausted: true` in the per-MLP report and `adjusted_final_layer_score_m = final_layer_mse_m × 1.0` for the affected MLP. There is also a **post-hoc** combined-budget check: even if flopscope didn't fire, the scoring layer checks `C_m = F_m + λ·R_m > flop_budget` after `predict()` returns and surfaces `combined_budget_exhausted: true` (same zero/×1.0 outcome).

## How do I inspect budget summaries while debugging?

Use:

- `budget.summary()` for the current explicit `BudgetContext`
- `flops.budget_summary()` for the accumulated process/session view
- `budget.summary_dict(...)` or `flops.budget_summary_dict(...)` for structured data

If you want namespace attribution, pass `by_namespace=True`.

## Is scoring hardware-dependent?

No. flopscope counts FLOPs analytically based on tensor shapes — not wall-clock time. The same estimator produces the same FLOP count on any hardware. You can develop on a laptop and submit for evaluation on a cluster with identical results.

## How many MLP networks are in a full evaluation?

The default evaluation scores your estimator on 10 MLPs (configured by `n_mlps` in `ContestSpec`). Each MLP has the same width and depth but different random weights and a distinct grader-supplied `mlp.seed` for any estimator-side randomness. Your aggregate score is the mean of the per-MLP `adjusted_final_layer_score` values.

## What if my estimator is fast but inaccurate?

You are ranked by the **budget-adjusted** `adjusted_final_layer_score = final_layer_mse × max(0.1, C_m / B)`, not raw MSE. Using less than 10% of the effective-compute budget gets you the 0.1 multiplier floor — a factor-of-ten discount and no more. So extremely cheap and inaccurate beats moderately cheap and inaccurate only up to that floor; below it, there is no further benefit to being cheaper.

## My local score is great but my submission scores 10x worse — why?

Almost always one of three things:

1. **Module-level state survives between predict() calls in-process.** Your Stage 3 (`--runner local`) iteration accidentally caches results between MLPs (lookup tables, RNG state, memoized partials). Stage 4 (`--runner subprocess`) and the grader run each MLP in a fresh process — that state is gone, and your score collapses. **Fix:** move state to instance attributes (`self._...`) populated in `setup()`, or use the `SetupContext.scratch_dir` for cross-call caching that's recomputed deterministically.

2. **Imports that work in-process fail in a clean subprocess.** A relative import, a missing `requirements.txt` entry, or a side-effecting top-level statement. **Fix:** run `uv run whest run --estimator estimator.py --runner subprocess` locally before submitting, and read the "Estimator Errors" panel.

3. **Numerical non-determinism without a seed.** Random MLP generation, Monte-Carlo ground truth, or your estimator's own RNG. **Fix:** add `--seed N` to your local runs to compare apples-to-apples, and avoid time-based seeds in your estimator.

If your Stage 3 and Stage 4 scores agree but the grader still disagrees, suspect Python-version or BLAS-version drift — `uv run whest doctor` will surface the relevant runtime info.

## ➡️ Next step

- [Common Participant Errors](./common-participant-errors.md)
- [Debugging Checklist](../how-to/debugging-checklist.md)
- [Scoring Model](../concepts/scoring-model.md)

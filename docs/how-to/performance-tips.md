# Performance Tips

> [← Documentation](../README.md)

This page lists concrete patterns for reducing FLOP usage in your estimator.

## Stay in float32

Since flopscope 0.9, every op's cost is also scaled by a `dtype_rate`: 1.0× for
32-bit-or-smaller dtypes, 2.0× for float64/int64 (up to 4.0× for float128).
NumPy promotion decides the billing dtype from your operands, so a single
stray float64 array upgrades an entire expression to the 2× rate, matmuls
included. `fnp.zeros()`/`fnp.ones()`/`fnp.eye()`/RNG draws all default to float64
when you don't pass `dtype=`. This is usually the single biggest lever on your
FLOP total: pass `dtype=fnp.float32` when you create arrays and don't need the
extra precision.

**`mlp.weights` arrive as float32.** whestbench 0.16.0 pins them on every
construction path (`domain.py` `MLP.from_row`, `generation.sample_mlp`), so
there is nothing to cast. Confirm it in one line:

```python
print(mlp.weights[0].dtype)   # float32
```

Casting anyway is not free: `w.astype(fnp.float32)` on an already-float32
`(1024, 1024)` weight still bills 1,048,576 FLOPs per layer, 16,777,216 across
the 16 layers, for no change at all.

**Seed your own state explicitly.** One `fnp.zeros(width)` with no `dtype=` is
float64, and NumPy promotion drags every matmul against it onto the 2× rate no
matter what the weights are.

Both bundled analytical baselines seed at float32 for exactly this reason, and
the effect is measured, not theoretical. At the competition shape
(`width=1024, depth=16`):

| Example | all-float64 † | float32 state | Saving | Final-layer MSE |
|---|---:|---:|---:|---|
| [`02_mean_propagation`](../../examples/02_mean_propagation.py) | 153,913,344 | **86,639,616** | 43.7% | unchanged to 5 s.f. |
| [`03_covariance_propagation`](../../examples/03_covariance_propagation.py) | 103,407,495,614 | **51,709,240,799** | 49.99% | unchanged to 5 s.f. |

Roughly half the bill, for free, on estimators whose accuracy does not move.

† The two counterfactuals are defined slightly differently. The `02` figure
drops the `dtype=fnp.float32` seeds **and** the two `.astype(fnp.float32)`
recasts after `flops.stats.norm` (the fully-float64 estimator); dropping the
seeds alone costs 153,978,880. The `03` figure drops only the seeds and keeps
its recasts.

**Watch for silent re-promotion.** `flops.stats.norm.cdf` / `.pdf` promote
float32 input to float64 to match `scipy.stats`, and flopscope emits a
`FlopscopeWarning` saying so. If you let the promoted result flow onward, it
re-infects the rest of the loop at the 2× rate and undoes the seeding. In
`02_mean_propagation` that alone was the difference between a 2.7% saving and a
43.7% one. Cast straight back:

```python
phi_alpha = flops.stats.norm.pdf(alpha).astype(fnp.float32)
Phi_alpha = flops.stats.norm.cdf(alpha).astype(fnp.float32)
```

`.astype()` bills 1 FLOP per element at the operand's dtype rate. Here the
input is the float64 result of `flops.stats.norm`, so 1024 elements × 2.0 =
2,048 FLOPs per call at width 1024, against the millions it protects. Check
`ctx.summary()` if you suspect a promotion: a `float64` row where you expected
`float32` is the tell.

## Residual wall time is a hard gate, not a currency

You are ranked on effective compute `C_m = F_m`. There is no λ term and no
wall-clock term: you cannot buy accuracy with seconds, and seconds do not cost
you FLOPs. What wall time does instead is gate the run. Residual wall time
(anything flopscope does not meter) is subject to a **hard cap of 400 ms per
MLP**, on top of the **120 s** wall-clock cap on `predict()` as a whole. Cross
either and that MLP's prediction is replaced by zeros: no partial credit, no
warning, and the MSE for that MLP is whatever a zero prediction earns.

Residual time exists for plumbing: unpacking `mlp`, control flow around your
`fnp` calls, assembling the array you return. **It is not a compute budget.**
Doing meaningful computation outside `flopscope.numpy` is prohibited, not
priced: the rules treat it as an attempt to evade FLOP accounting.

Concretely, all of the following are **prohibited**, and using them is grounds
for disqualifying the submission. They are not a residual-time expense you can
choose to pay:

- Vendoring or bundling your own `numpy`, `scipy`, or any BLAS.
- Compiled kernels of any kind, and `ctypes` / `cffi` / any other FFI.
- `threading`, `asyncio`, `subprocess`, or `multiprocessing`.
- Running your own computation while a flopscope op is in flight.
- Touching the flopscope client, its transport, or its accounting.

Your submission may use the flopscope client API and the pure-Python standard
library, and nothing else. Shipping **data** files (weights, lookup tables,
other precomputed artifacts) remains permitted; it is *computation* outside
flopscope that is not. See
[`examples/04_shipped_weights.py`](../../examples/04_shipped_weights.py) for the
supported way to bring precomputed work with you.

Staying comfortably inside the 400 ms cap:

- **Vectorise per-neuron Python loops into `fnp` array ops.** A 1024-iteration
  Python loop is the most common way to blow the residual cap, and it is doing
  in slow Python exactly the work flopscope would have metered for you.
- **Don't tune an internal deadline to your own machine's clock.** A
  `time.time()` cutoff calibrated locally can trip early elsewhere and return
  your fallback answer with no error. See
  [Is scoring hardware-dependent?](../troubleshooting/faq.md#is-scoring-hardware-dependent).
- **Measure it:** `uv run whest run --estimator estimator.py --profile`, then
  read `residual_wall_time_s` per MLP. `--residual-wall-time-limit` already
  defaults to 0.4, the graded cap, so a plain run gates you exactly as the live
  round does. Pass the flag only to make the local run *stricter*. Anything
  approaching 400 ms locally is a failure waiting to happen on hardware you
  have not seen.
- **Stress the wall clock too:** re-running with `--max-threads 1` pins the BLAS
  pool so `wall_time_s` is comparable across machines. It is a pessimistic
  bound rather than a prediction, but it is a cheap way to see whether you are
  anywhere near the 120 s cap.

If you have a legitimate reason to sit near the residual cap (plumbing that
genuinely needs the time, not computation), ask before you submit:
[arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com).

## Matmul dominates your budget

A single `fnp.matmul(A, B)` on two (n, n) matrices costs O(n^3) FLOPs, exactly `n · n · (2n − 1)`, and float64 costs exactly 2× float32 on top of that; see [Stay in float32](#stay-in-float32) above. Because the cost is cubic in the width, the competition width of 1024 makes matrix work dominate everything else in your estimator. Even [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py), which never does a matrix-matrix product at all, still spends 77.42% of its 86,639,616 FLOPs in `matmul`. See the [worked walkthrough](./manage-flop-budget.md#worked-walkthrough-mean-propagation-line-by-line).

**Tip:** If you only need diagonal information (per-neuron variance), avoid full matrix-matrix multiplies. Diagonal propagation uses matrix-vector products: O(n^2) per layer instead of O(n^3).

## Free operations — use them liberally

These cost 0 FLOPs in flopscope:

- `fnp.zeros()`, `fnp.empty()`
- Views: `.T` / `fnp.transpose()`, basic indexing and slicing (`x[0]`, `x[:, 3]`)
- `fnp.diag()`, `fnp.diagonal()`
- `fnp.asarray()` on an existing flopscope array (no copy), `fnp.random.default_rng(seed)` construction

**Since flopscope 0.9, these look free but aren't; they bill 1×/element (2× at float64):** `fnp.ones()`, `fnp.eye()`, `fnp.full()`, `fnp.array()` (and any copying `asarray`), `.copy()`, `.astype()`, `fnp.reshape()`/`.reshape()` (billed even where NumPy itself would return a view), `fnp.concatenate()`, `fnp.stack()`, `tile()`, `repeat()`. Cheap next to a matmul, but no longer zero. Don't reach for them as a "free" replacement for real computation.

Precompute anything you can using the still-free ops above; for the billed ones, computing once outside a per-layer loop still beats recomputing every iteration. There is no separate memory cost in FLOP terms: flopscope only meters compute, not allocation. Memory is not unlimited, though: the solution process gets **8 GB**, so a batch sized purely by the FLOP budget can still run you out of RAM.

## Precompute outside the layer loop

If your estimator computes something that does not change per-layer, move it before the loop:

```python
import flopscope.numpy as fnp

# Instead of this (wasteful):
for w in mlp.weights:
    scale = fnp.sqrt(2.0 / mlp.width)  # recomputed every layer
    ...

# Do this (computed once):
scale = fnp.sqrt(2.0 / mlp.width)  # computed once, ~2 FLOPs total — not free, but trivial
for w in mlp.weights:
    ...
```

## Diagonal vs full covariance — an accuracy call, not a budget call

At the competition shape (`width=1024, depth=16`) both bundled baselines fit the
per-MLP budget of 2,199,023,255,552 FLOPs with room to spare:

| Approach | Cost per layer | Measured per `predict()` | Share of the per-MLP budget |
|----------|---------------|-------------------------:|----------------------------:|
| Mean propagation (diagonal) | O(width^2) | 86,639,616 | 0.004% |
| Covariance propagation (full) | O(width^3) | 51,709,240,799 | 2.351% |

Full covariance is 597× the cost of mean propagation and *still* uses under 3%
of the budget. The score multiplier is `max(0.1, C_m / B_m)`, so anything below
10% of the budget leaves the multiplier pinned at its floor. Spending more
FLOPs there costs you nothing. Pick the approach that predicts best, and only
start trading accuracy for FLOPs once you are pushing past that floor.

## Check your budget breakdown

Use `flops.budget_summary()` inside a `BudgetContext` to see exactly where your FLOPs go:

```python
import flopscope as flops

with flops.BudgetContext(flop_budget=2_199_023_255_552) as budget:
    result = estimator.predict(mlp, budget=2_199_023_255_552)
    flops.budget_summary()
```

This prints a per-operation table showing call counts and cumulative FLOPs. Look for the dominant operation and optimize that first.

## Skip hardware fallback probes during local iteration

If startup latency matters while you are iterating locally, you can skip the extra OS-native hardware fallback probes that populate report and dataset metadata:

```bash
WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1 uv run whest run --estimator estimator.py
```

This keeps cheap metadata collection and `psutil`-backed fields enabled. Only the fallback probes are skipped, so fields such as `cpu_count_physical` or `ram_total_bytes` may remain `null` when they are not already available.

## ➡️ Next step

- [Manage Your FLOP Budget](./manage-flop-budget.md)
- [Algorithm Ideas](./algorithm-ideas.md)
- [Code Patterns](../reference/code-patterns.md)

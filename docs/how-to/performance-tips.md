# Performance Tips

> [← Documentation](../README.md)

This page lists concrete patterns for reducing FLOP usage in your estimator.

## Stay in float32

Since flopscope 0.9, every op's cost is also scaled by a `dtype_rate`: 1.0× for
32-bit-or-smaller dtypes, 2.0× for float64/int64 (up to 4.0× for float128).
NumPy promotion decides the billing dtype from your operands, so a single
stray float64 array upgrades an entire expression to the 2× rate — including
matmuls. `fnp.zeros()`/`fnp.ones()`/RNG draws all default to float64 when you
don't pass `dtype=`. This is usually the single biggest lever on your FLOP
total: pass `dtype=fnp.float32` when you create arrays and don't need the
extra precision.

## Matmul dominates your budget

A single `fnp.matmul(A, B)` on two (n, n) matrices costs O(n^3) FLOPs — exactly `n · n · (2n − 1)`. For width=256 **in float32**, that is ~33M FLOPs per matmul (33,488,896). In a 32-layer network, 32 matmuls cost ~1.07B FLOPs — well within the 2.72e11 default budget, but the cost dominates for any moderately-sized estimator. The same matmuls in float64 cost exactly 2× (~67M/matmul, ~2.14B for the network) — see [Stay in float32](#stay-in-float32) above.

**Tip:** If you only need diagonal information (per-neuron variance), avoid full matrix-matrix multiplies. Diagonal propagation uses matrix-vector products: O(n^2) per layer instead of O(n^3).

## Free operations — use them liberally

These cost 0 FLOPs in flopscope:

- `fnp.zeros()`, `fnp.empty()`
- Views: `.T` / `fnp.transpose()`, basic indexing and slicing (`x[0]`, `x[:, 3]`)
- `fnp.diag()`, `fnp.diagonal()`
- `fnp.asarray()` on an existing flopscope array (no copy), `fnp.random.default_rng(seed)` construction

**Since flopscope 0.9, these look free but aren't — they bill 1×/element (2× at float64):** `fnp.ones()`, `fnp.eye()`, `fnp.full()`, `fnp.array()` (and any copying `asarray`), `.copy()`, `.astype()`, `fnp.reshape()`/`.reshape()` (billed even where NumPy itself would return a view), `fnp.concatenate()`, `fnp.stack()`, `tile()`, `repeat()`. Cheap next to a matmul, but no longer zero — don't reach for them as a "free" replacement for real computation.

Precompute anything you can using the still-free ops above; for the billed ones, computing once outside a per-layer loop still beats recomputing every iteration. There is no separate memory cost in FLOP terms — flopscope only meters compute, not allocation.

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

## Diagonal vs full covariance — know when to switch

| Approach | Cost per layer | When to use |
|----------|---------------|-------------|
| Mean propagation (diagonal) | O(width^2) | Default. Budget < 30 x width^2 |
| Covariance propagation (full) | O(width^3) | Budget >= 30 x width^2 |

## Check your budget breakdown

Use `flops.budget_summary()` inside a `BudgetContext` to see exactly where your FLOPs go:

```python
import flopscope as flops

with flops.BudgetContext(flop_budget=272_000_000_000) as budget:
    result = estimator.predict(mlp, budget=272_000_000_000)
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

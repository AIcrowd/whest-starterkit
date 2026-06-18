# Performance Tips

> [← Documentation](../README.md)

This page lists concrete patterns for reducing FLOP usage in your estimator.

## Matmul dominates your budget

A single `fnp.matmul(A, B)` on two (n, n) matrices costs O(n^3) FLOPs. For width=256, that is ~33M FLOPs per matmul. In a 32-layer network, 32 matmuls cost ~1.07B FLOPs — well within the 2.72e11 default budget, but the cost dominates for any moderately-sized estimator.

**Tip:** If you only need diagonal information (per-neuron variance), avoid full matrix-matrix multiplies. Diagonal propagation uses matrix-vector products: O(n^2) per layer instead of O(n^3).

## Free operations — use them liberally

These cost 0 FLOPs in flopscope:

- `fnp.zeros()`, `fnp.ones()`, `fnp.eye()`, `fnp.array()`
- `fnp.reshape()`, `fnp.transpose()`
- `fnp.concatenate()`, `fnp.stack()`
- Indexing: `x[0]`, `x[:, 3]`, `fnp.diag(M)`

Precompute anything you can using free ops. Store intermediate values in variables — there is no memory cost in FLOP terms.

## Precompute outside the layer loop

If your estimator computes something that does not change per-layer, move it before the loop:

```python
import flopscope.numpy as fnp

# Instead of this (wasteful):
for w in mlp.weights:
    scale = fnp.sqrt(2.0 / mlp.width)  # recomputed every layer
    ...

# Do this (free):
scale = fnp.sqrt(2.0 / mlp.width)  # computed once
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

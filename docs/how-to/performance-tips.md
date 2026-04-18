# Performance Tips

This page lists concrete patterns for reducing FLOP usage in your estimator.

## Matmul dominates your budget

A single `we.matmul(A, B)` on two (n, n) matrices costs O(n^3) FLOPs. For width=100, that is ~1M FLOPs per matmul. In a 16-layer network, 16 matmuls cost ~16M FLOPs — a large fraction of a typical budget.

**Tip:** If you only need diagonal information (per-neuron variance), avoid full matrix-matrix multiplies. Diagonal propagation uses matrix-vector products: O(n^2) per layer instead of O(n^3).

## Free operations — use them liberally

These cost 0 FLOPs in whest:

- `we.zeros()`, `we.ones()`, `we.eye()`, `we.array()`
- `we.reshape()`, `we.transpose()`
- `we.concatenate()`, `we.stack()`
- Indexing: `x[0]`, `x[:, 3]`, `we.diag(M)`

Precompute anything you can using free ops. Store intermediate values in variables — there is no memory cost in FLOP terms.

## Precompute outside the layer loop

If your estimator computes something that does not change per-layer, move it before the loop:

```python
import whest as we

# Instead of this (wasteful):
for w in mlp.weights:
    scale = we.sqrt(2.0 / mlp.width)  # recomputed every layer
    ...

# Do this (free):
scale = we.sqrt(2.0 / mlp.width)  # computed once
for w in mlp.weights:
    ...
```

## Diagonal vs full covariance — know when to switch

| Approach | Cost per layer | When to use |
|----------|---------------|-------------|
| Mean propagation (diagonal) | O(width^2) | Default. Budget < 30 x width^2 |
| Covariance propagation (full) | O(width^3) | Budget >= 30 x width^2 |

The combined estimator in `examples/estimators/combined_estimator.py` shows this routing pattern.

## Check your budget breakdown

Use `we.budget_summary()` inside a `BudgetContext` to see exactly where your FLOPs go:

```python
import whest as we

with we.BudgetContext(flop_budget=100_000_000) as budget:
    result = estimator.predict(mlp, budget=100_000_000)
    we.budget_summary()
```

This prints a per-operation table showing call counts and cumulative FLOPs. Look for the dominant operation and optimize that first.

## Skip hardware fallback probes during local iteration

If startup latency matters while you are iterating locally, you can skip the extra OS-native hardware fallback probes that populate report and dataset metadata:

```bash
WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1 whest run --estimator ./my_estimator.py
```

This keeps cheap metadata collection and `psutil`-backed fields enabled. Only the fallback probes are skipped, so fields such as `cpu_count_physical` or `ram_total_bytes` may remain `null` when they are not already available.

## Next step

- [Manage Your FLOP Budget](./manage-flop-budget.md)
- [Algorithm Ideas](./algorithm-ideas.md)
- [Code Patterns](../reference/code-patterns.md)

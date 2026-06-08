# Manage Your FLOP Budget

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page to understand how FLOP budgets work and how to optimize your estimator to stay within budget.

## Why FLOPs, not wall-clock time

This challenge scores estimators by **analytical FLOP count**, not execution time. Every mathematical operation your estimator performs is tracked by [flopscope](https://github.com/AIcrowd/flopscope) — a NumPy-compatible library that counts floating-point operations deterministically from tensor shapes.

This means your score is **hardware-independent**: the same estimator produces the same FLOP count on a laptop and a GPU cluster. You can focus on algorithmic efficiency rather than hardware tuning.

For the full flopscope API and cost model, see the [flopscope documentation](https://github.com/AIcrowd/flopscope).

## Which operations cost FLOPs

| Category | Examples | Cost |
|----------|----------|------|
| **Free (0 FLOPs)** | `fnp.array()`, `fnp.zeros()`, `fnp.ones()`, `fnp.reshape()`, `fnp.transpose()`, indexing, `fnp.concatenate()`, `fnp.stack()` | No budget impact |
| **Pointwise (1 FLOP/element)** | `fnp.add()`, `fnp.multiply()`, `fnp.exp()`, `fnp.sqrt()`, `fnp.maximum()` | Output element count |
| **Reductions** | `fnp.sum()`, `fnp.mean()`, `fnp.max()` | Input element count |
| **Matrix operations** | `fnp.matmul()`, `fnp.einsum()` | Depends on dimensions — typically dominates your budget |
| **Random samplers** | `rng.standard_normal()`, `rng.uniform()` (where `rng = fnp.random.default_rng(seed)`); same for module-level `fnp.random.standard_normal()` etc. and `fnp.random.RandomState(seed)` | Calibrated per method (default ~16 FLOPs/element for `standard_normal`) |

**Key insight:** `fnp.matmul` on `(n, n)` matrices costs `O(n^3)` FLOPs. For width-256 networks, a single matmul costs ~33M FLOPs. Most of your budget goes to matrix operations.

## Check your budget usage

Wrap your estimator logic in a `BudgetContext` to see how many FLOPs it consumes:

```python
import flopscope as flops

with flops.BudgetContext(flop_budget=68_000_000_000) as budget:
    result = estimator.predict(mlp, budget=68_000_000_000)

print(f"FLOPs used: {budget.flops_used:,}")
print(f"FLOPs remaining: {budget.flops_remaining:,}")
```

If you also want a wall-clock guardrail while debugging locally, set
`wall_time_limit_s` on the same `BudgetContext`:

```python
with flops.BudgetContext(
    flop_budget=68_000_000_000,
    wall_time_limit_s=2.0,
) as budget:
    result = estimator.predict(mlp, budget=68_000_000_000)
```

## Get a per-operation breakdown

Use `budget.summary()` for the current explicit context or
`flops.budget_summary()` for the session/global view to see which operations
consume the most FLOPs:

```python
import flopscope as flops

with flops.BudgetContext(flop_budget=68_000_000_000) as budget:
    result = estimator.predict(mlp, budget=68_000_000_000)
    print(budget.summary())

flops.budget_summary()
```

This prints a table showing each operation's name, call count, and cumulative FLOP cost — letting you identify the expensive operations to optimize.

The same summaries also show timing data:

- `wall_time_s`: total elapsed time for the context
- `flopscope_backend_time_s`: time spent inside counted flopscope backend calls
- `flopscope_overhead_time_s`: time spent inside flopscope dispatch and bookkeeping
- `residual_wall_time_s`: time spent outside flopscope backend and dispatch work

In `whest run`, the CLI flags map to these concepts as follows:

- `--wall-time-limit`: forwards a wall-clock limit into the estimator's `BudgetContext`
- `--residual-wall-time-limit`: adds a WhestBench scoring check on the reported `residual_wall_time_s`

## Interpret `whest run` output

When you run your estimator with `whest run`, the per-MLP report includes:

- **`flops_used`**: total FLOPs your estimator consumed for that MLP.
- **`budget_exhausted`**: `true` if your estimator exceeded the FLOP budget — predictions were zeroed.
- **`final_layer_mse`** / **`all_layers_mse`**: your prediction accuracy (lower is better).

If `budget_exhausted` is `true`, your predictions were discarded. You need to reduce FLOP usage.

## Worked walkthrough: mean propagation, line by line

The table below profiles [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py) on the default Stage 1 MLP (`width=256, depth=8` — same shape as the Stage-3 grader). Numbers are aggregated across all 8 layers; per-layer cost is roughly the row total divided by 8. Reproduce with `ctx.summary()` inside a `flopscope.BudgetContext` after a single `predict()` call (profiled under flopscope 0.5.0).

| Operation in `predict()` | Calls | FLOPs (total) | % of `predict()` total |
|---|---:|---:|---:|
| `mu_pre = w.T @ mu` and `var_pre = (w*w).T @ var` (`matmul`) | 16 | 2,093,056 | **77.1%** |
| `mu_pre * Phi_alpha + sigma_pre * phi_alpha` etc. (`multiply`) | 64 | 538,624 | 19.8% |
| `flops.stats.norm.pdf(alpha)` | 8 | 32,768 | 1.2% |
| `flops.stats.norm.cdf(alpha)` | 8 | 32,768 | 1.2% |
| `mu_pre * Phi_alpha + ...` etc. (`add`) | 24 | 6,144 | 0.2% |
| `fnp.maximum(var_pre, 1e-12)` (`maximum`) | 16 | 4,096 | 0.2% |
| `fnp.sqrt(var_pre)` | 8 | 2,048 | 0.1% |
| `mu_pre / sigma_pre` (`true_divide`) | 8 | 2,048 | 0.1% |
| `ez2 - mu*mu` (`subtract`) | 8 | 2,048 | 0.1% |
| `fnp.stack(rows, axis=0)` | 1 | 2,048 | 0.1% |
| **Total per `predict()`** | — | **2,715,648** | — |

The full ~2.7 M FLOPs spends only ~0.004% of the default 6.8e10 grader budget, so mean propagation lands well below the multiplier floor at this shape — see [Scoring Model](../concepts/scoring-model.md#example-estimator-benchmarks).

Two takeaways:

- **`matmul` dominates.** ~77% of `predict()` cost is the two matmuls per layer (the pointwise ReLU-moment terms — `multiply` — are the visible ~20% remainder). Halving the matmul count (e.g., switching to a diagonal-only formulation, or fusing into a single `einsum` like `examples/03_covariance_propagation.py` does for the symmetric cov-update) buys you most of that back.
- **Reductions, sqrt, and divides are free in practice.** Don't twist your code to avoid them; the cost is in the tens of FLOPs per layer.

The same pattern holds for `examples/03_covariance_propagation.py`, where the `O(width³)` symmetry-aware `einsum` lands at ~404 M FLOPs per `predict()` (~0.6% of the grader budget) — ~150× more expensive than mean propagation (its full covariance is genuinely heavier than mean propagation's diagonal variance), but still leaving plenty of headroom.

## Optimization tips

1. **Matmul dominates.** Each `fnp.matmul(W.T, mu)` on a `(width, width)` matrix costs `O(width^2)` FLOPs per layer. Reducing the number of matmuls (or their dimensions) has the biggest impact.

2. **Diagonal approximations save FLOPs.** Mean propagation uses diagonal variance (`O(width^2)` per layer) instead of full covariance propagation (`O(width^3)` per layer). Choose the right level of approximation for your budget.

3. **Array creation is free.** `fnp.array()`, `fnp.zeros()`, `fnp.ones()`, `fnp.eye()` cost 0 FLOPs. Precompute and store intermediate values freely.

4. **Pick one strategy per estimator.** Use either mean propagation or full covariance as your default implementation, then optimize it for the fixed budget.

## ➡️ Next step

- [Write an Estimator](./write-an-estimator.md)
- [Scoring Model](../concepts/scoring-model.md)
- [Profile Simulation](../advanced/profile-simulation.md)
- [Estimator Contract](../reference/estimator-contract.md)

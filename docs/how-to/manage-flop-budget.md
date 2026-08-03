# Manage Your FLOP Budget

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page to understand how FLOP budgets work and how to optimize your estimator to stay within budget.

## Why FLOPs, not wall-clock time

This challenge scores estimators by **analytical FLOP count**, not execution time. Every mathematical operation your estimator performs is tracked by [flopscope](https://github.com/AIcrowd/flopscope) — a NumPy-compatible library that counts floating-point operations deterministically from tensor shapes and dtypes.

This means your **FLOP count** is hardware-independent: the same estimator produces the same `F_m` on a laptop and on the grader. Your *score* is not purely `F_m`, though — you are ranked on effective compute `C_m = F_m + λ·R_m`, and `R_m` (residual wall time) is measured on the grader's hardware, which the rules do not guarantee will match yours. Keep the math inside `flopscope.numpy` and `R_m` stays negligible; push work outside it and you are being timed. See [Is scoring hardware-dependent?](../troubleshooting/faq.md#is-scoring-hardware-dependent).

For the full flopscope API and cost model, see the [flopscope documentation](https://github.com/AIcrowd/flopscope).

## Which operations cost FLOPs

| Category | Examples | Cost |
|----------|----------|------|
| **Free (0 FLOPs)** | `fnp.zeros()`, `fnp.empty()`, views (`.T`/`fnp.transpose()`, basic slicing), `fnp.asarray()` on an existing flopscope array (no copy), `fnp.random.default_rng(seed)` construction | No budget impact |
| **Pointwise (1 FLOP/element)** | `fnp.add()`, `fnp.multiply()`, `fnp.sqrt()`, `fnp.maximum()`, comparisons — **and data movement that used to be free**: `fnp.ones()`/`full()`/`eye()` fills, `fnp.array()`/copying `fnp.asarray()`, `.copy()`, `.astype()`, `reshape()`/`ravel()`, `fnp.stack()`, `fnp.concatenate()`, `tile()`, `repeat()` | Output element count |
| **Reductions** | `fnp.sum()`, `fnp.mean()`, `fnp.max()` | Input element count |
| **Transcendental (16 FLOP/element)** | `fnp.exp()`, `fnp.log()`, trig, and `x ** y` power (including `x ** 2`) | Output element count × 16 |
| **Matrix operations** | `fnp.matmul()`, `fnp.einsum()` | Depends on dimensions — typically dominates your budget |
| **Random samplers** | `rng.standard_normal()`, `rng.uniform()` (where `rng = fnp.random.default_rng(seed)`); same for module-level `fnp.random.standard_normal()` etc. and `fnp.random.RandomState(seed)` | Calibrated per method (`standard_normal`: 16 FLOPs/element at float32, **32/element at the float64 default**) |

**Key insights:**

- `fnp.matmul` on `(n, n)` matrices costs `O(n^3)` FLOPs. For width-256 networks, a single matmul costs ~33M FLOPs (float32; float64 doubles it). Most of your budget goes to matrix operations.
- Every cost above is also scaled by a `dtype_rate`: 1.0× for 32-bit-or-smaller dtypes, 2.0× for float64/int64 (up to 4.0× for float128). NumPy promotion decides the billing dtype from your operands, so one stray float64 array upgrades the whole expression — see the [flopscope primer](../reference/flopscope-primer.md#operation-flop-costs) for the full weight/dtype-rate table.

## Check your budget usage

Wrap your estimator logic in a `BudgetContext` to see how many FLOPs it consumes:

```python
import flopscope as flops

with flops.BudgetContext(flop_budget=272_000_000_000) as budget:
    result = estimator.predict(mlp, budget=272_000_000_000)

print(f"FLOPs used: {budget.flops_used:,}")
print(f"FLOPs remaining: {budget.flops_remaining:,}")
```

If you also want a wall-clock guardrail while debugging locally, set
`wall_time_limit_s` on the same `BudgetContext`:

```python
with flops.BudgetContext(
    flop_budget=272_000_000_000,
    wall_time_limit_s=2.0,
) as budget:
    result = estimator.predict(mlp, budget=272_000_000_000)
```

## Get a per-operation breakdown

Use `budget.summary()` for the current explicit context or
`flops.budget_summary()` for the session/global view to see which operations
consume the most FLOPs:

```python
import flopscope as flops

with flops.BudgetContext(flop_budget=272_000_000_000) as budget:
    result = estimator.predict(mlp, budget=272_000_000_000)
    print(budget.summary())

flops.budget_summary()
```

This prints a table showing each operation's name, call count, and cumulative FLOP cost — letting you identify the expensive operations to optimize.

The same summaries also show timing data:

- `wall_time_s`: total elapsed time for the context
- `flopscope_backend_time_s`: time spent inside counted flopscope backend calls
- `flopscope_overhead_time_s`: time spent inside flopscope dispatch and bookkeeping
- `residual_wall_time_s`: participant Python (loops, control flow), GC, and Python-callback op time; as of flopscope 0.7.0, data-movement NumPy ops (concatenate, stack, tile, repeat, take, pad, …) count as `flopscope_backend_time_s`, not residual

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

The table below profiles [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py) on the phase-1 competition shape (`width=256, depth=32`; the warmup round used 256×8). Numbers are aggregated across all 32 layers; per-layer cost is roughly the row total divided by 32. Reproduce with `ctx.summary()` inside a `flopscope.BudgetContext` after a single `predict()` call (profiled under flopscope 0.10.0).

| Operation in `predict()` | Calls | FLOPs (total) | % of `predict()` total |
|---|---:|---:|---:|
| `mu_pre = w.T @ mu` and `var_pre = (w*w).T @ var` (`matmul`) | 64 | 16,744,448 | **82.4%** |
| `mu_pre * Phi_alpha + sigma_pre * phi_alpha` etc. (`multiply`) | 256 | 2,211,840 | 10.9% |
| `flops.stats.norm.cdf(alpha)` | 32 | 786,432 | 3.9% |
| `flops.stats.norm.pdf(alpha)` | 32 | 442,368 | 2.2% |
| `mu_pre * Phi_alpha + ...` etc. (`add`) | 96 | 49,152 | 0.2% |
| `fnp.maximum(var_pre, 1e-12)` (`maximum`) | 64 | 32,768 | 0.2% |
| `fnp.sqrt(var_pre)` | 32 | 16,384 | 0.1% |
| `mu_pre / sigma_pre` (`true_divide`) | 32 | 16,384 | 0.1% |
| `ez2 - mu*mu` (`subtract`) | 32 | 16,384 | 0.1% |
| `fnp.stack(rows, axis=0)` | 1 | 16,384 | 0.1% |
| `var = fnp.ones(width)` (`ones`) | 1 | 512 | 0.0% |
| `mu = fnp.zeros(width)` (`zeros`) | 1 | 0 | 0.0% |
| **Total per `predict()`** | — | **20,333,056** | — |

The full ~20.3 M FLOPs spends only ~0.0075% of the 2.72e11 grader budget, so mean propagation lands well below the multiplier floor at this shape — see [Scoring Model](../concepts/scoring-model.md#example-estimator-benchmarks).

Two takeaways:

- **`matmul` dominates.** ~82% of `predict()` cost is the two matmuls per layer (the pointwise ReLU-moment terms — `multiply` — are the visible ~11% remainder). Halving the matmul count (e.g., switching to a diagonal-only formulation, or fusing into a single `einsum` like `examples/03_covariance_propagation.py` does for the symmetric cov-update) buys you most of that back.
- **Sqrt, divides, and clamps stay cheap.** Don't twist your code to avoid them: `sqrt`, `true_divide`, and `maximum` each cost ~512 FLOPs per call here (256 elements × the float64 dtype rate this walkthrough happens to run at, since `mu`/`var` start from `fnp.zeros(width)`/`fnp.ones(width)` with no explicit `dtype=`) — trivial next to the matmuls.

The same pattern holds for `examples/03_covariance_propagation.py`, where the `O(width³)` symmetry-aware `einsum` — plus the per-layer `as_symmetric()` re-validation that keeps its symmetric rate under flopscope ≥ 0.10.0 — lands at ~3.3 B FLOPs per `predict()` (~1.2% of the grader budget, re-profiled the same way under flopscope 0.10.0) — ~160× more expensive than mean propagation (its full covariance is genuinely heavier than mean propagation's diagonal variance, and it inherits the same float64 default), but still leaving plenty of headroom.

## Optimization tips

1. **Matmul dominates.** Each `fnp.matmul(W.T, mu)` on a `(width, width)` matrix costs `O(width^2)` FLOPs per layer. Reducing the number of matmuls (or their dimensions) has the biggest impact.

2. **Diagonal approximations save FLOPs.** Mean propagation uses diagonal variance (`O(width^2)` per layer) instead of full covariance propagation (`O(width^3)` per layer). Choose the right level of approximation for your budget.

3. **Only `fnp.zeros()`/`fnp.empty()` and no-copy views are free.** Since flopscope 0.9, `fnp.array()`, `fnp.ones()`, and `fnp.eye()` each bill 1 FLOP per element at float32 (2× that at the float64 default) — `eye` bills only its diagonal length, not the full matrix. They're still cheap next to any matmul, but calling them inside a hot loop is no longer free — prefer `fnp.zeros()` or a no-copy `fnp.asarray()` for placeholders you'll overwrite anyway.

4. **Pick one strategy per estimator.** Use either mean propagation or full covariance as your default implementation, then optimize it for the fixed budget.

## ➡️ Next step

- [Write an Estimator](./write-an-estimator.md)
- [Scoring Model](../concepts/scoring-model.md)
- [Profile Simulation](../advanced/profile-simulation.md)
- [Estimator Contract](../reference/estimator-contract.md)

# Manage Your FLOP Budget

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page to understand how FLOP budgets work and how to optimize your estimator to stay within budget.

## Why FLOPs, not wall-clock time

This challenge scores estimators by **analytical FLOP count**, not execution time. Every mathematical operation your estimator performs is tracked by [flopscope](https://github.com/AIcrowd/flopscope) — a NumPy-compatible library that counts floating-point operations deterministically from tensor shapes and dtypes.

This means your **FLOP count** is hardware-independent: the same estimator produces the same `F_m` on a laptop and on the grader. Your effective compute is `C_m = F_m` — nothing else is priced into your score, so you cannot trade wall-clock time for FLOPs or FLOPs for wall-clock time.

Wall time is a *gate*, not a currency. Residual wall time — anything flopscope does not meter — is capped at **400 ms per MLP**, and `predict()` as a whole is capped at **120 s per MLP**. Cross either one and that MLP's prediction is replaced by zeros. Residual time exists for plumbing (unpacking `mlp`, control flow around your `fnp` calls, assembling the returned array), not for computation: doing real work outside `flopscope.numpy` is **prohibited**, not merely expensive. See [Performance Tips](./performance-tips.md#residual-wall-time-is-a-hard-gate-not-a-currency) and [Is scoring hardware-dependent?](../troubleshooting/faq.md#is-scoring-hardware-dependent).

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

- `fnp.matmul` on `(n, n)` matrices costs `O(n^3)` FLOPs; a matrix-vector product costs `O(n^2)`. The cost is cubic in the width, so at the Phase 2 width of 1024 matrix work dominates everything else — even in an estimator that only ever does matrix-vector products, `matmul` is 77.42% of the total (see the [worked walkthrough](#worked-walkthrough-mean-propagation-line-by-line) below).
- Every cost above is also scaled by a `dtype_rate`: 1.0× for 32-bit-or-smaller dtypes, 2.0× for float64/int64 (up to 4.0× for float128). NumPy promotion decides the billing dtype from your operands, so one stray float64 array upgrades the whole expression — see the [flopscope primer](../reference/flopscope-primer.md#operation-flop-costs) for the full weight/dtype-rate table.

## Check your budget usage

The Phase 2 per-MLP budget is `B_m = 2**41 = 2,199,023,255,552` FLOPs. Wrap your estimator logic in a `BudgetContext` with that budget to see how many FLOPs it consumes:

```python
import flopscope as flops

with flops.BudgetContext(flop_budget=2_199_023_255_552) as budget:
    result = estimator.predict(mlp, budget=2_199_023_255_552)

print(f"FLOPs used: {budget.flops_used:,}")
print(f"FLOPs remaining: {budget.flops_remaining:,}")
```

If you also want a wall-clock guardrail while debugging locally, set
`wall_time_limit_s` on the same `BudgetContext`:

```python
with flops.BudgetContext(
    flop_budget=2_199_023_255_552,
    wall_time_limit_s=2.0,
) as budget:
    result = estimator.predict(mlp, budget=2_199_023_255_552)
```

## Get a per-operation breakdown

Use `budget.summary()` for the current explicit context or
`flops.budget_summary()` for the session/global view to see which operations
consume the most FLOPs:

```python
import flopscope as flops

with flops.BudgetContext(flop_budget=2_199_023_255_552) as budget:
    result = estimator.predict(mlp, budget=2_199_023_255_552)
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

- `--wall-time-limit`: forwards a wall-clock limit into the estimator's `BudgetContext`. The grader enforces 120 s per MLP.
- `--residual-wall-time-limit`: adds a WhestBench scoring check on the reported `residual_wall_time_s`. The grader enforces a hard 400 ms per MLP, so `--residual-wall-time-limit 0.4` is the setting that matches the live round.

## Interpret `whest run` output

When you run your estimator with `whest run`, the per-MLP report includes:

- **`flops_used`**: total FLOPs your estimator consumed for that MLP.
- **`budget_exhausted`**: `true` if your estimator exceeded the FLOP budget — predictions were zeroed.
- **`residual_wall_time_s`** / **`residual_wall_time_exhausted`**: time spent outside metered flopscope ops, and whether it crossed the limit. On the grader the limit is 400 ms per MLP and crossing it zeroes that MLP's predictions.
- **`final_layer_mse`** / **`all_layers_mse`**: your prediction accuracy (lower is better).

If `budget_exhausted` is `true`, your predictions were discarded. You need to reduce FLOP usage. If `residual_wall_time_exhausted` is `true`, the problem is not FLOPs at all — see [Performance Tips](./performance-tips.md#residual-wall-time-is-a-hard-gate-not-a-currency).

## Worked walkthrough: mean propagation, line by line

The table below profiles [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py) on the competition shape (`width=1024, depth=16`). Numbers are aggregated across all 16 layers; per-layer cost is roughly the row total divided by 16. Reproduce with `ctx.summary()` inside a `flopscope.BudgetContext` after a single `predict()` call (profiled under flopscope 0.11.0, the version this kit pins; identical under 0.12.0).

| Operation in `predict()` | Calls | FLOPs (total) | % of `predict()` total |
|---|---:|---:|---:|
| `mu_pre = w.T @ mu` and `var_pre = (w*w).T @ var` (`matmul`) | 32 | 67,076,096 | **77.42%** |
| `mu_pre * Phi_alpha + sigma_pre * phi_alpha` etc. (`multiply`) | 128 | 16,891,904 | 19.50% |
| `flops.stats.norm.cdf(alpha)` | 16 | 1,572,864 | 1.82% |
| `flops.stats.norm.pdf(alpha)` | 16 | 884,736 | 1.02% |
| `.astype(fnp.float32)` on the cdf/pdf results (`astype`) | 32 | 65,536 | 0.08% |
| `mu_pre * Phi_alpha + ...` etc. (`add`) | 48 | 49,152 | 0.06% |
| `fnp.maximum(var_pre, 1e-12)` (`maximum`) | 32 | 32,768 | 0.04% |
| `fnp.sqrt(var_pre)` | 16 | 16,384 | 0.02% |
| `mu_pre / sigma_pre` (`true_divide`) | 16 | 16,384 | 0.02% |
| `ez2 - mu*mu` (`subtract`) | 16 | 16,384 | 0.02% |
| `fnp.stack(rows, axis=0)` | 1 | 16,384 | 0.02% |
| `var = fnp.ones(width, dtype=fnp.float32)` (`ones`) | 1 | 1,024 | 0.00% |
| `mu = fnp.zeros(width, dtype=fnp.float32)` (`zeros`) | 1 | 0 | 0.00% |
| **Total per `predict()`** | — | **86,639,616** | — |

The full ~86.6 M FLOPs spends only ~0.004% of the 2,199,023,255,552-FLOP grader budget, so mean propagation lands well below the multiplier floor at this shape — see [Scoring Model](../concepts/scoring-model.md#example-estimator-benchmarks).

Three takeaways:

- **`matmul` dominates.** ~77% of `predict()` cost is the two matmuls per layer (the pointwise ReLU-moment terms — `multiply` — are the visible ~20% remainder). Halving the matmul count (e.g., switching to a diagonal-only formulation, or fusing into a single `einsum` like `examples/03_covariance_propagation.py` does for the symmetric cov-update) buys you most of that back.
- **dtype is worth more than any single op here.** This walkthrough profiles the example *as shipped*, which seeds `mu`/`var` at `dtype=fnp.float32`. Leave that off and everything the two arrays touch — the matmuls included — bills at the float64 2× rate: the same estimator costs **153,913,344** FLOPs, 43.7% more, for a final-layer MSE identical to five significant figures. `mlp.weights` is already float32, so the float64 buys nothing. See [Stay in float32](./performance-tips.md#stay-in-float32).
- **Sqrt, divides, and clamps stay cheap.** Don't twist your code to avoid them: `sqrt`, `true_divide`, and `maximum` each cost 1,024 FLOPs per call here (1024 elements at the float32 rate) — trivial next to the matmuls.

The same pattern holds for `examples/03_covariance_propagation.py`, where the `O(width³)` symmetry-aware `einsum` — plus the per-layer `as_symmetric()` re-validation that keeps its symmetric rate under flopscope ≥ 0.10.0 — lands at ~51.7 B FLOPs per `predict()` (51,709,240,799 exactly; 2.351% of the grader budget, profiled the same way under flopscope 0.11.0) — 597× more expensive than mean propagation, because a full covariance is genuinely heavier than a diagonal variance. It seeds at float32 too; leaving the default would put it at 103,407,495,614 — a 99.98% increase, not quite a doubling, because the 32 `stats.norm` calls bill float64 either way.

## Optimization tips

1. **Matmul dominates.** Each `fnp.matmul(W.T, mu)` on a `(width, width)` matrix costs `O(width^2)` FLOPs per layer. Reducing the number of matmuls (or their dimensions) has the biggest impact.

2. **Diagonal approximations save FLOPs.** Mean propagation uses diagonal variance (`O(width^2)` per layer) instead of full covariance propagation (`O(width^3)` per layer) — 86,639,616 FLOPs against 51,709,240,799 at this shape. Note that both sit under the score's multiplier floor of `max(0.1, C_m / B_m)`: below 10% of the budget the multiplier does not move, so extra compute that buys accuracy is effectively free up to that point. Choose the level of approximation that maximises accuracy, not the cheapest one.

3. **Only `fnp.zeros()`/`fnp.empty()` and no-copy views are free.** Since flopscope 0.9, `fnp.array()`, `fnp.ones()`, and `fnp.eye()` each bill 1 FLOP per element at float32 (2× that at the float64 default) — `eye` bills only its diagonal length, not the full matrix. They're still cheap next to any matmul, but calling them inside a hot loop is no longer free — prefer `fnp.zeros()` or a no-copy `fnp.asarray()` for placeholders you'll overwrite anyway.

4. **Pick one strategy per estimator.** Use either mean propagation or full covariance as your default implementation, then optimize it for the fixed `2**41` budget.

5. **Don't try to move work out of the FLOP budget.** Computation performed outside `flopscope.numpy` — bundled numpy or BLAS, compiled kernels, FFI, threads or subprocesses — is prohibited and grounds for disqualification, not a cheaper way to buy accuracy. See [Performance Tips](./performance-tips.md#residual-wall-time-is-a-hard-gate-not-a-currency).

If you believe flopscope is mispricing an operation, report it at [arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com) rather than building around it.

## ➡️ Next step

- [Write an Estimator](./write-an-estimator.md)
- [Scoring Model](../concepts/scoring-model.md)
- [Profile Simulation](../advanced/profile-simulation.md)
- [Estimator Contract](../reference/estimator-contract.md)

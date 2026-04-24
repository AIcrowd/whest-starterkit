# Manage Your FLOP Budget

## When to use this page

Use this page to understand how FLOP budgets work and how to optimize your estimator to stay within budget.

## Why FLOPs, not wall-clock time

This challenge scores estimators by **analytical FLOP count**, not execution time. Every mathematical operation your estimator performs is tracked by [whest](https://github.com/AIcrowd/whest) — a NumPy-compatible library that counts floating-point operations deterministically from tensor shapes.

This means your score is **hardware-independent**: the same estimator produces the same FLOP count on a laptop and a GPU cluster. You can focus on algorithmic efficiency rather than hardware tuning.

For the full whest API and cost model, see the [whest documentation](https://github.com/AIcrowd/whest).

## Which operations cost FLOPs

| Category | Examples | Cost |
|----------|----------|------|
| **Free (0 FLOPs)** | `we.array()`, `we.zeros()`, `we.ones()`, `we.reshape()`, `we.transpose()`, indexing, `we.concatenate()`, `we.stack()` | No budget impact |
| **Pointwise (1 FLOP/element)** | `we.add()`, `we.multiply()`, `we.exp()`, `we.sqrt()`, `we.maximum()` | Output element count |
| **Reductions** | `we.sum()`, `we.mean()`, `we.max()` | Input element count |
| **Matrix operations** | `we.matmul()`, `we.einsum()` | Depends on dimensions — typically dominates your budget |
| **Random generation** | `we.random.normal()`, `we.random.uniform()` | Output element count |

**Key insight:** `we.matmul` on `(n, n)` matrices costs `O(n^3)` FLOPs. For width-100 networks, a single matmul costs ~1M FLOPs. Most of your budget goes to matrix operations.

## Check your budget usage

Wrap your estimator logic in a `BudgetContext` to see how many FLOPs it consumes:

```python
import whest as we

with we.BudgetContext(flop_budget=100_000_000) as budget:
    result = estimator.predict(mlp, budget=100_000_000)

print(f"FLOPs used: {budget.flops_used:,}")
print(f"FLOPs remaining: {budget.flops_remaining:,}")
```

If you also want a wall-clock guardrail while debugging locally, set
`wall_time_limit_s` on the same `BudgetContext`:

```python
with we.BudgetContext(
    flop_budget=100_000_000,
    wall_time_limit_s=2.0,
) as budget:
    result = estimator.predict(mlp, budget=100_000_000)
```

## Get a per-operation breakdown

Use `budget.summary()` for the current explicit context or
`we.budget_summary()` for the session/global view to see which operations
consume the most FLOPs:

```python
import whest as we

with we.BudgetContext(flop_budget=100_000_000) as budget:
    result = estimator.predict(mlp, budget=100_000_000)
    print(budget.summary())

we.budget_summary()
```

This prints a table showing each operation's name, call count, and cumulative FLOP cost — letting you identify the expensive operations to optimize.

The same summaries also show timing data:

- `wall_time_s`: total elapsed time for the context
- `tracked_time_s`: time spent inside counted whest calls
- `untracked_time_s`: time spent outside counted whest calls

In `whest run`, the CLI flags map to these concepts as follows:

- `--wall-time-limit`: forwards a wall-clock limit into the estimator's `BudgetContext`
- `--untracked-time-limit`: adds a WhestBench scoring check on the reported `untracked_time_s`

## Interpret `whest run` output

When you run your estimator with `whest run`, the per-MLP report includes:

- **`flops_used`**: total FLOPs your estimator consumed for that MLP.
- **`budget_exhausted`**: `true` if your estimator exceeded the FLOP budget — predictions were zeroed.
- **`final_mse`** / **`all_layer_mse`**: your prediction accuracy (lower is better).

If `budget_exhausted` is `true`, your predictions were discarded. You need to reduce FLOP usage.

## Optimization tips

1. **Matmul dominates.** Each `we.matmul(W.T, mu)` on a `(width, width)` matrix costs `O(width^2)` FLOPs per layer. Reducing the number of matmuls (or their dimensions) has the biggest impact.

2. **Diagonal approximations save FLOPs.** Mean propagation uses diagonal variance (`O(width^2)` per layer) instead of full covariance propagation (`O(width^3)` per layer). Choose the right level of approximation for your budget.

3. **Array creation is free.** `we.array()`, `we.zeros()`, `we.ones()`, `we.eye()` cost 0 FLOPs. Precompute and store intermediate values freely.

4. **Use the combined estimator pattern.** Route between cheap (mean propagation) and expensive (covariance propagation) algorithms based on the available FLOP budget. See [`examples/estimators/combined_estimator.py`](../../examples/estimators/combined_estimator.py).

## Next step

- [Write an Estimator](./write-an-estimator.md)
- [Scoring Model](../concepts/scoring-model.md)
- [Profile Simulation](./profile-simulation.md)
- [Estimator Contract](../reference/estimator-contract.md)

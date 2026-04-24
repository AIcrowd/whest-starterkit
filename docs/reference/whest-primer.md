# Whest Primer

Whest is a numpy-compatible array library that tracks FLOPs analytically rather than timing them on hardware. Every arithmetic operation on a `we.ndarray` increments a FLOP counter instead of (or in addition to) performing the computation. This is how WhestBench enforces fair FLOP budgets across different machines.

Source: [github.com/AIcrowd/whest](https://github.com/AIcrowd/whest)

## BudgetContext

All estimator predictions run inside a `BudgetContext`. When the budget is exhausted, a `BudgetExhaustedError` is raised and your predictions are zeroed out.

```python
import whest as we

with we.BudgetContext(flop_budget=1_000_000) as ctx:
    x = we.ones(100)
    y = x @ we.eye(100)  # matmul: 100 * 100 * 100 = 1M FLOPs
    # BudgetExhaustedError raised here if budget exceeded
```

You don't need to create `BudgetContext` yourself — the framework does it before calling your `predict()` method. The `budget` argument tells you how many FLOPs you have.

`BudgetContext` also supports `wall_time_limit_s` when you want a cooperative
wall-clock limit in addition to the FLOP cap:

```python
with we.BudgetContext(flop_budget=1_000_000, wall_time_limit_s=2.0) as ctx:
    ...
```

The timer starts when the context is entered and is checked before and after
each counted whest/NumPy call. If it is exceeded, whest raises
`TimeExhaustedError`.

## Operation FLOP Costs

| Category | Operations | Cost |
|----------|-----------|------|
| **Free** (0 FLOPs) | `we.array`, `we.zeros`, `we.ones`, `we.eye`, `we.asarray`, `we.reshape`, `.T`, indexing, `we.stack`, `we.concatenate`, `.copy()`, `.astype()` | 0 |
| **Pointwise** (1 FLOP/element) | `+`, `-`, `*`, `/`, `we.exp`, `we.sqrt`, `we.abs`, `we.maximum`, `we.where`, `we.log`, comparisons | N elements |
| **Reductions** (input size) | `we.sum`, `we.mean`, `we.var`, `we.max`, `we.min`, `we.all`, `we.any` | N elements |
| **Matmul** | `@`, `we.matmul` | M * N * K for (M,N) @ (N,K) |

**Key insight:** Matmul dominates. A single `(100, 100) @ (100, 100)` costs 1M FLOPs. A pointwise `exp` on 100 elements costs 100 FLOPs.

## Array Creation

```python
import whest as we

x = we.zeros(100)                          # 1D zeros
X = we.zeros((64, 100), dtype=we.float32)  # 2D zeros, explicit dtype
I = we.eye(100, dtype=we.float32)          # identity matrix
a = we.array([1.0, 2.0, 3.0])             # from list
b = we.asarray(numpy_array)                # convert from numpy (free)
```

All array creation is **free** (0 FLOPs).

## Random Number Generation

```python
import whest as we

rng = we.random.default_rng(42)            # seeded RNG
x = rng.standard_normal((1000, 64))        # Gaussian samples
x = x.astype(we.float32)                   # cast to float32 (free)
```

Random generation itself is free. FLOPs are counted when you operate on the arrays.

## Budget Inspection

Use `budget.summary()` for the current explicit context and
`we.budget_summary()` for the accumulated session/global view:

```python
with we.BudgetContext(flop_budget=10_000_000) as ctx:
    # ... your computations ...
    print(ctx.summary())        # current context only
    print(we.budget_summary())  # process/session-wide summary
    print(ctx.flops_used)       # integer FLOP count
```

Both summaries can also include timing data:

- `wall_time_s`: total elapsed time in the context
- `tracked_time_s`: time spent inside counted whest calls
- `untracked_time_s`: everything else in the context

This is useful during development to understand where both FLOPs and time go.

## WhestBench-specific limits

Whest itself only knows about `wall_time_limit_s` on `BudgetContext`.
WhestBench adds two run-level knobs on top:

- `--wall-time-limit`: passed through to the estimator's `BudgetContext`
- `--untracked-time-limit`: enforced by WhestBench after `predict()` returns,
  using the reported `untracked_time_s`

So if you see `untracked_time_exhausted` in a report, that came from
WhestBench scoring logic, not from a `BudgetContext` parameter.

## Common Gotchas

**numpy arrays still count FLOPs.** Since `we.ndarray` is backed by numpy, a raw numpy array passed to whest operations will still be tracked. Use `we.array()` or `we.asarray()` to convert explicitly.

**Pythonic operators are tracked.** `x @ w` counts the same FLOPs as `we.matmul(x, w)`. Use whichever reads better.

**dtype matters for precision, not FLOPs.** `float32` and `float64` operations cost the same FLOPs. Use `float32` for memory efficiency and `float64` for numerical stability where needed.

## Testing

Use whest's testing utilities:

```python
import whest as we

we.testing.assert_allclose(actual, expected, atol=1e-6)
we.testing.assert_array_equal(actual, expected)
```

These work like numpy's testing functions but on whest arrays.

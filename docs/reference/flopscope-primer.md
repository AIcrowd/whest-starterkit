# Flopscope Primer

> [← Documentation](../README.md)

Flopscope is a numpy-compatible array library that tracks FLOPs analytically rather than timing them on hardware. Every arithmetic operation on a `fnp.ndarray` increments a FLOP counter instead of (or in addition to) performing the computation. This is how WhestBench enforces fair FLOP budgets across different machines.

Source: [github.com/AIcrowd/flopscope](https://github.com/AIcrowd/flopscope)

## BudgetContext

All estimator predictions run inside a `BudgetContext`. When the budget is exhausted, a `BudgetExhaustedError` is raised and your predictions are zeroed out.

```python
import flopscope as flops
import flopscope.numpy as fnp

with flops.BudgetContext(flop_budget=1_000_000) as ctx:
    x = fnp.ones(100)
    y = x @ fnp.eye(100)  # matmul: 100 * 100 * 100 = 1M FLOPs
    # BudgetExhaustedError raised here if budget exceeded
```

You don't need to create `BudgetContext` yourself — something else opens it for you, and your `predict()` body runs inside that scope. Who that "something else" is depends on which stage you're in:

| Stage | Who opens the `BudgetContext` | Where to look |
|---|---|---|
| 1 — `python estimator.py` | `local_engine.compare_against_monte_carlo` (default `estimator_budget=4e9`) | [local_engine.py](../../local_engine.py) |
| 2 — `whest validate` | the validator (small probe budget on a width=4, depth=2 MLP) | the `whestbench` CLI |
| 3 — `whest run --runner local` | the in-process harness (default `--flop-budget 2.72e11`) | the `whestbench` CLI |
| 4 — `whest run --runner subprocess` | the subprocess worker (same default) | the `whestbench` CLI |
| Grader (after you submit) | the harness inside the grader's sandboxed container | (runs server-side on AIcrowd) |

The `budget` integer your `predict(mlp, budget)` receives matches the
`flop_budget` of the surrounding context and is the hard cap for that call.
or ignore it if you always run the same strategy.

`BudgetContext` also supports `wall_time_limit_s` when you want a cooperative
wall-clock limit in addition to the FLOP cap:

```python
with flops.BudgetContext(flop_budget=1_000_000, wall_time_limit_s=2.0) as ctx:
    ...
```

The timer starts when the context is entered and is checked before and after
each counted flopscope/NumPy call. If it is exceeded, flopscope raises
`TimeExhaustedError`.

## Operation FLOP Costs

| Category | Operations | Cost |
|----------|-----------|------|
| **Free** (0 FLOPs) | `fnp.array`, `fnp.zeros`, `fnp.ones`, `fnp.eye`, `fnp.asarray`, `fnp.reshape`, `.T`, indexing, `fnp.stack`, `fnp.concatenate`, `.copy()`, `.astype()`, `fnp.random.default_rng(seed)` (constructing the RNG) | 0 |
| **Pointwise** (1 FLOP/element) | `+`, `-`, `*`, `/`, `fnp.exp`, `fnp.sqrt`, `fnp.abs`, `fnp.maximum`, `fnp.where`, `fnp.log`, comparisons | N elements |
| **Reductions** (input size) | `fnp.sum`, `fnp.mean`, `fnp.var`, `fnp.max`, `fnp.min`, `fnp.all`, `fnp.any` | N elements |
| **Random samplers** | `rng.standard_normal(n)`, `rng.uniform(...)`, `fnp.random.standard_normal(...)` and module-level analogs; same for `RandomState(seed)` | calibrated per method (default ~16 FLOPs/element for `standard_normal`; lower weights for cheap samplers like `uniform`) |
| **Matmul** | `@`, `fnp.matmul` | M * N * K for (M,N) @ (N,K) |

**Key insight:** Matmul dominates. A single `(100, 100) @ (100, 100)` costs 1M FLOPs. A pointwise `exp` on 100 elements costs 100 FLOPs.

## Array Creation

```python
import flopscope as flops
import flopscope.numpy as fnp

x = fnp.zeros(100)                          # 1D zeros
X = fnp.zeros((64, 100), dtype=fnp.float32)  # 2D zeros, explicit dtype
I = fnp.eye(100, dtype=fnp.float32)          # identity matrix
a = fnp.array([1.0, 2.0, 3.0])             # from list
b = fnp.asarray(numpy_array)                # convert from numpy (free)
```

All array creation is **free** (0 FLOPs).

## Random Number Generation

```python
import flopscope as flops
import flopscope.numpy as fnp

rng = fnp.random.default_rng(42)            # seeded RNG (free)
x = rng.standard_normal((1000, 64))         # ~64,000 × 16 FLOPs charged
x = x.astype(fnp.float32)                   # cast to float32 (free)
```

Random samplers **are FLOP-counted** ([flopscope#81](https://github.com/AIcrowd/flopscope/pull/81)):
`rng.standard_normal(...)`, `rng.uniform(...)`, and the module-level
analogs (`fnp.random.standard_normal(...)`, etc.) all deduct from the
active `BudgetContext` and return `FlopscopeArray`. The same holds for
`fnp.random.RandomState(seed)`. Constructing the RNG itself is free;
only the sampling methods cost FLOPs. Cross-API parity is guaranteed —
the three idioms above all charge the same FLOPs for the same physical
sample count.

Per-method weights are calibrated empirically (default ~16 FLOPs per
element for `standard_normal`; cheaper methods like `uniform` have
lower weights). See `flopscope.numpy.random._registry` upstream for the
authoritative table.

## Budget Inspection

Inside an active `BudgetContext`, the `ctx` object exposes the following
public attributes and methods. Most are only useful while debugging or
profiling — your `predict()` body usually only needs the `budget` integer
that the harness passed in.

| Attribute | Type | Meaning |
|---|---|---|
| `flop_budget` | `int` | Cap configured at construction time. |
| `flops_used` | `int` | Total counted FLOPs since the context was entered. |
| `flops_remaining` | `int` | `flop_budget - flops_used`. |
| `wall_time_s` | `float` | Elapsed wall time since the context was entered. |
| `wall_time_limit_s` | `float \| None` | Cap configured at construction time. |
| `flopscope_backend_time_s` | `float` | Time spent inside counted flopscope calls. |
| `flopscope_overhead_time_s` | `float` | Time spent inside flopscope's own dispatch (wrapper preambles, FLOP bookkeeping, namespace push/pop) — framework cost, not participant cost. |
| `residual_wall_time_s` | `float` | Wall time inside the context that is neither flopscope backend execution nor flopscope's own dispatch — i.e. participant Python (loops, control flow) and GC. As of flopscope 0.7.0, data-movement NumPy ops (concatenate, stack, tile, repeat, take, pad, …) are counted as `flopscope_backend_time_s`, not residual; Python-callback ops bill their callback time here. |
| `elapsed_s` | `float` | Alias of `wall_time_s` for symmetry with the report. |
| `namespace` | `str \| None` | Namespace this context attributes ops to (set via `with flops.namespace("name")`). |
| `op_log` | `list[OpRecord]` | Per-op record (only populated under `--profile`). |
| `summary()` | method | Pretty-printed summary for the current context. |
| `summary_dict(...)` | method | Same data as a `dict` (machine-readable). |
| `deduct(n)` | method | Manually attribute `n` FLOPs to this context (use sparingly — flopscope's instrumentation handles the common cases). |

```python
with flops.BudgetContext(flop_budget=10_000_000) as ctx:
    # ... your computations ...
    print(ctx.flops_used, "/", ctx.flop_budget)   # quick check
    print(ctx.flops_remaining)
    print(ctx.summary())                          # rich per-op breakdown
    print(flops.budget_summary())                 # process/session-wide
```

The session-wide `flops.budget_summary()` and `flops.budget_summary_dict()`
aggregate across every context entered in the current Python process —
useful when you're profiling a multi-stage pipeline.

Both summaries also include four timing fields that satisfy this strict
timing identity:

```text
wall_time_s = flopscope_backend_time_s + flopscope_overhead_time_s + residual_wall_time_s
```

- `wall_time_s`: total elapsed time in the context
- `flopscope_backend_time_s`: time spent inside counted flopscope numpy kernels (the participant's actual numpy compute)
- `flopscope_overhead_time_s`: time spent inside flopscope's own dispatch (wrapper preambles, FLOP bookkeeping, namespace push/pop) — framework cost, not participant cost
- `residual_wall_time_s`: participant Python (loops, control flow), GC, and Python-callback op time; as of flopscope 0.7.0, data-movement NumPy ops (concatenate, stack, tile, repeat, take, pad, …) count as `flopscope_backend_time_s`, not residual

This decomposition lets you see whether time is going to numpy compute, framework dispatch, or your own Python.

## WhestBench-specific limits

Flopscope's `BudgetContext` measures `wall_time_s`, `flopscope_backend_time_s`,
`flopscope_overhead_time_s`, and `residual_wall_time_s`. It also accepts
`wall_time_limit_s`, which it checks while counted flopscope operations run.

WhestBench exposes some of those concepts as run-level CLI knobs:

- `--wall-time-limit`: passed through to the estimator's `BudgetContext`
- `--residual-wall-time-limit`: enforced by WhestBench after `predict()` returns,
  using the reported `residual_wall_time_s`. Because `residual_wall_time_s`
  excludes flopscope backend and dispatch time, this gate measures only your
  Python and uninstrumented work — not numpy backend execution or the
  framework's bookkeeping tax.

So if you see `time_exhausted`, that came from Flopscope's `wall_time_limit_s`.
If you see `residual_wall_time_exhausted`, that came from WhestBench scoring
logic comparing Flopscope's measured `residual_wall_time_s` with the configured
`--residual-wall-time-limit`.

## Common Gotchas

**numpy arrays still count FLOPs.** Since `fnp.ndarray` is backed by numpy, a raw numpy array passed to flopscope operations will still be tracked. Use `fnp.array()` or `fnp.asarray()` to convert explicitly.

**Pythonic operators are tracked.** `x @ w` counts the same FLOPs as `fnp.matmul(x, w)`. Use whichever reads better.

**dtype matters for precision, not FLOPs.** `float32` and `float64` operations cost the same FLOPs. Use `float32` for memory efficiency and `float64` for numerical stability where needed.

## Testing

Use flopscope's testing utilities:

```python
import flopscope as flops
import flopscope.numpy as fnp

fnp.testing.assert_allclose(actual, expected, atol=1e-6)
fnp.testing.assert_array_equal(actual, expected)
```

These work like numpy's testing functions but on flopscope arrays.

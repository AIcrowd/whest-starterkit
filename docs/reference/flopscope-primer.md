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
    x = fnp.ones((100, 100), dtype=fnp.float32)  # fill: 10,000 FLOPs (billed since 0.9)
    w = fnp.ones((100, 100), dtype=fnp.float32)  # fill: 10,000 FLOPs (billed since 0.9)
    y = x @ w                                     # matmul: 100·100·(2·100−1) ≈ 2M FLOPs
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

As of flopscope 0.9, every operation is billed as

```
charged = flop_cost × weight × dtype_rate × complex_factor
```

- `flop_cost` — the op's shape-derived count (e.g. `M·N·(2K−1)` for a `(M,K) @ (K,N)` matmul: K multiplies + K−1 adds per output element).
- `weight` — the op's tier, one of **{0, 1, 4, 16}**.
- `dtype_rate` — **1.0 for 32-bit-or-smaller dtypes, 2.0 for float64/int64, up to 4.0 for float128**. The billing dtype follows NumPy promotion over the operands, so one stray float64 vector makes the whole expression bill 2×.
- `complex_factor` — real-FLOP equivalents for complex dtypes (e.g. complex multiply = 6).

| Tier | Operations | Cost at float32 |
|------|-----------|------|
| **Free** (weight 0) | `fnp.zeros`, `fnp.empty`, views (`.T`, basic slicing), `fnp.asarray` on an existing flopscope array (no copy), `fnp.random.default_rng(seed)` construction | 0 |
| **1× per element** | `+`, `-`, `*`, `/`, comparisons, `fnp.maximum`, `fnp.sqrt`, reductions (`sum`, `mean`, `var`, `max`, …), **and data movement that used to be free**: `fnp.ones`/`full`/`eye` fills, `fnp.array`/`asarray` copies, `.copy()`, `.astype()`, `reshape`/`ravel` (billed even when NumPy would return a view), `stack`, `concatenate`, `tile`, `repeat` | N elements (`eye`: diagonal length written) |
| **4× per element** | Gathers (`take`, fancy `arr[int_idx]`), value-ordering (`sort`, `argsort`, `unique`, `searchsorted`), 3-arg `fnp.where` | 4·N |
| **16× per element** | Transcendentals: `fnp.exp`, `fnp.log`, trig, and **`x ** y` power — including `x ** 2`** | 16·N |
| **Matmul** | `@`, `fnp.matmul`, `fnp.einsum` | `M·N·(2K−1)` for `(M,K) @ (K,N)` |
| **Random samplers** | `rng.standard_normal(...)` 16/element at float32 (32 at the float64 default), `rng.uniform(...)` 6/element (always float64 — no `dtype=` arg; calibrated float32 base rate 3) | calibrated per method |

**Key insights:**
- Matmul still dominates: `(100,100) @ (100,100)` in float32 costs ~2M FLOPs (`M·N·(2K−1)` = 1,990,000).
- **Stay in float32.** The same code on float64 arrays bills exactly 2×. `fnp.zeros(n)` defaults to float64 — pass `dtype=fnp.float32` when you don't need the precision.
- **Write `x * x`, not `x ** 2`** — power routes through the 16× transcendental tier.
- Composite helpers bill their internals: `flops.stats.norm.pdf` ≈ 54/element and `flops.stats.norm.cdf` ≈ 96/element.

The weight and rate tables above are a summary. The audited, authoritative
per-op reference (including complex factors, accumulator-widening rules for
integer reductions, and per-family formulas) is flopscope's
[cost model reference](https://aicrowd.github.io/flopscope/docs/reference/cost-model)
— and `ctx.summary()` on your own run is always ground truth.
`tests/test_flopscope_cost_docs.py` in this kit pins the claims made on this
page so a future flopscope bump flags them.

## Array Creation

```python
import flopscope as flops
import flopscope.numpy as fnp

x = fnp.zeros(100)                           # 1D zeros — free (defaults to float64)
X = fnp.zeros((64, 100), dtype=fnp.float32)  # 2D zeros, explicit dtype — free
I = fnp.eye(100, dtype=fnp.float32)          # identity: 100 FLOPs (diagonal only, billed since 0.9)
a = fnp.array([1.0, 2.0, 3.0])               # from list: 6 FLOPs (3 elems, float64 default doubles the rate)
b = fnp.asarray(numpy_array)                 # convert from numpy — free (no copy needed)
```

Since flopscope 0.9, only `fnp.zeros` / `fnp.empty` and no-copy views are free.
Fills (`ones`, `eye`, `full`) and copies (`fnp.array`, copying `asarray`,
`.astype()`, `.copy()`) bill 1× per element written — small next to any matmul,
but no longer zero.

## Random Number Generation

```python
import flopscope as flops
import flopscope.numpy as fnp

rng = fnp.random.default_rng(42)            # seeded RNG (free)
x = rng.standard_normal((1000, 64))         # 64,000 × 32 FLOPs (float64 default)
x32 = rng.standard_normal((1000, 64), dtype=fnp.float32)  # 64,000 × 16 FLOPs — draw in f32 when you can
```

Random samplers **are FLOP-counted** ([flopscope#81](https://github.com/AIcrowd/flopscope/pull/81)):
`rng.standard_normal(...)`, `rng.uniform(...)`, and the module-level
analogs (`fnp.random.standard_normal(...)`, etc.) all deduct from the
active `BudgetContext` and return `FlopscopeArray`. The same holds for
`fnp.random.RandomState(seed)`. Constructing the RNG itself is free;
only the sampling methods cost FLOPs. Cross-API parity is guaranteed —
the three idioms above all charge the same FLOPs for the same physical
sample count.

Per-method weights are calibrated empirically (16 FLOPs per element for
float32 `standard_normal`, 32 at the float64 default; cheaper samplers
like `uniform` bill 6/element (always float64 — `uniform` has no `dtype=`
switch; its calibrated float32 base rate is 3)). See
`flopscope.numpy.random._registry` upstream for the authoritative table.

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
| `op_log` | `list[OpRecord]` | Per-op record (only populated under `--profile`). Each `OpRecord` carries `resolved_dtype` — which dtype priced the op. |
| `summary()` | method | Pretty-printed summary for the current context. |
| `summary_dict(...)` | method | Same data as a `dict` (machine-readable). |
| `deduct(op_name, *, flop_cost, shapes, dtypes, …)` | method | Manually attribute FLOPs. The signature changed in flopscope 0.9: name the op and pass `flop_cost` plus the operand `dtypes` — the dtype rate is applied on top (e.g. `ctx.deduct("my_op", flop_cost=10, subscripts=None, shapes=(), dtypes=(fnp.float64,))` charges 20). Pass `dtypes=()` for a dtype-neutral charge; `dtypes=None` raises. |

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

**dtype now matters for FLOPs too.** Since flopscope 0.9, float64 operations
bill 2× float32 (`dtype_rate`). NumPy promotion decides the billing dtype, so
one float64 operand upgrades the whole expression. Keep estimator state in
float32 unless you need the precision — and note `fnp.zeros(...)` defaults to
float64. Exotic dtypes without a billing rate raise `UnsupportedDtypeError`
(a `TypeError`) before any FLOPs are charged.

**Shape-only cost estimators assume float32.** The `flops.accounting.*`
helpers (`einsum_cost`, `svd_cost`, …) price at the float32 anchor; the same
op on float64 arrays bills 2× the estimate at runtime.

## Testing

Use flopscope's testing utilities:

```python
import flopscope as flops
import flopscope.numpy as fnp

fnp.testing.assert_allclose(actual, expected, atol=1e-6)
fnp.testing.assert_array_equal(actual, expected)
```

These work like numpy's testing functions but on flopscope arrays.

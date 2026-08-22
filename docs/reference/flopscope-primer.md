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
    # -> BudgetExhaustedError: matmul would cost 1,990,000 FLOPs but only 980,000 remain
```

That budget is deliberately too small: the two fills cost 10,000 each and the
matmul 1,990,000, for 2,010,000 against a 1,000,000 cap. Raise it to
`10_000_000` if you want the snippet to run; `ctx.flops_used` then reads
`2010000`.

You don't need to create `BudgetContext` yourself. Something else opens it for you, and your `predict()` body runs inside that scope. Who that "something else" is depends on which stage you're in:

| Stage | Who opens the `BudgetContext` | Where to look |
|---|---|---|
| 1 — `python estimator.py` | `local_engine.compare_against_monte_carlo` (default `estimator_budget=2**41`) | [local_engine.py](../../local_engine.py) |
| 2 — `whest validate` | **nobody** — validate calls `predict()` outside any `BudgetContext`, under flopscope's implicit 1e15 process default. It hands you `budget=100` on a width=4, depth=2 MLP as a shape/finiteness smoke test only; FLOPs are **not** capped here, so validate will never surface a budget bust. | the `whestbench` CLI |
| 3 — `whest run --runner local` | the in-process harness (`--flop-budget`) | the `whestbench` CLI |
| 4 — `whest run --runner subprocess` | the subprocess worker (same flag) | the `whestbench` CLI |
| Grader (after you submit) | the harness inside the grader's sandboxed container | (runs server-side on AIcrowd) |

> **The per-MLP budget is `2**41` = `2,199,023,255,552` FLOPs.** Since
> whestbench 0.16.0 that is the `whest run` default for `--flop-budget`, so
> stages 3 and 4 already meter you at the graded cap with no flag. Stage 1 uses
> the same number by default. Pass `--flop-budget` only to *lower* it
> deliberately, or to reproduce an earlier round (`272000000000` for
> v1-phase1, `68000000000` for v1-warmup).

In stages 1, 3 and 4 the `budget` integer your `predict(mlp, budget)` receives
matches the `flop_budget` of the surrounding context and is the hard cap for
that call. In stage 2 it does not: `whest validate` hands you `100` purely as
a value to branch on and enforces nothing. Branch on it if you have both a
cheap and an expensive strategy, or ignore it if you always run the same one.

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
| **1× per element** | `+`, `-`, `*`, `/`, comparisons, `fnp.maximum`, `fnp.sqrt`, **and data movement that used to be free**: `fnp.ones`/`full`/`eye` fills, `fnp.array`/`asarray` copies, `.copy()`, `.astype()`, `reshape`/`ravel` (billed even when NumPy would return a view), `stack`, `concatenate`, `tile`, `repeat` | N elements (`eye`: diagonal length written) |
| **Reductions** | `sum`/`max`/`min`/`prod`/`all`/`any`/`argmax`/`cumsum` | N−1 |
| | `mean`/`median`/`average` | N |
| | **`var`** / **`std`** | **4·N** / **4·N+1** |
| **4× per element** | Gathers (`take`, fancy `arr[int_idx]`), 3-arg `fnp.where` | 4·N |
| **Sorting/search** (weight 4, per comparison) | `sort`, `argsort`, `unique`, `searchsorted` | ≈4·N·⌈log₂N⌉ |
| **16× per element** | Transcendentals: `fnp.exp`, `fnp.log`, trig, and **`x ** y` power — including `x ** 2`** | 16·N |
| **Matmul** | `@`, `fnp.matmul`, `fnp.einsum` | `M·N·(2K−1)` for `(M,K) @ (K,N)` |
| **Random samplers** | `rng.standard_normal(...)` 16/element at float32 (32 at the float64 default), `rng.uniform(...)` 6/element (always float64 — no `dtype=` arg; calibrated float32 base rate 3) | calibrated per method |

**In practice:**
- Matmul still dominates: `(100,100) @ (100,100)` in float32 costs ~2M FLOPs (`M·N·(2K−1)` = 1,990,000).
- **Stay in float32.** The same code on float64 arrays bills exactly 2×. `fnp.zeros(n)` defaults to float64 — pass `dtype=fnp.float32` when you don't need the precision.
- **Write `x * x`, not `x ** 2`** — power routes through the 16× transcendental tier.
- **`var` costs 4× what `mean` costs** (4,000 vs 1,000 FLOPs on 1,000 float32 elements), and since flopscope 0.12.0 most `nan*` reductions add an N-element NaN scan on top of their plain counterpart: roughly 2× on the cheap ones (`nansum` 1,999 vs `sum` 999), proportionally less on the dear ones (`nanvar` 5,000 vs `var` 4,000), and nothing at all on `nanmax`/`nanmin` (999 either way). Drop the `nan` prefix when your data cannot contain NaNs. Tiers are weights, not costs; the cost formula is separate, which is why `var` sits in the 1× tier and still bills 4·N.
- Composite helpers bill their internals: `flops.stats.norm.pdf` ≈ 54/element and `flops.stats.norm.cdf` ≈ 96/element.

The weight and rate tables above are a summary. The audited, authoritative
per-op reference (including complex factors, accumulator-widening rules for
integer reductions, and per-family formulas) is flopscope's
[cost model reference](https://aicrowd.github.io/flopscope/docs/understanding/flop-counting-model/).
`ctx.summary()` on your own run is always ground truth.
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
`.astype()`, `.copy()`) bill 1× per element written, small next to any matmul
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
only the sampling methods cost FLOPs. Cross-API parity is guaranteed:
the three idioms above all charge the same FLOPs for the same physical
sample count.

Per-method weights are calibrated empirically (16 FLOPs per element for
float32 `standard_normal`, 32 at the float64 default; cheaper samplers
like `uniform` bill 6/element, always float64, since `uniform` has no
`dtype=` switch and its calibrated float32 base rate is 3). See
`flopscope/data/default_weights.json` (the `weights` map: `random.standard_normal`
is 16.0, `random.uniform` is 1.0 over a 3·N cost formula) plus
`flopscope/numpy/random/_cost_formulas.py` for the shape formulas. Both ship
inside the installed wheel:

```bash
uv run python -c "import json, pathlib, flopscope; w=json.loads((pathlib.Path(flopscope.__file__).parent/'data/default_weights.json').read_text())['weights']; print({k: v for k, v in w.items() if k.startswith('random.')})"
```

## Budget Inspection

Inside an active `BudgetContext`, the `ctx` object exposes the following
public attributes and methods. Most are only useful while debugging or
profiling; your `predict()` body usually only needs the `budget` integer
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
| `op_log` | `list[OpRecord]` | Per-op record, populated in **every** `BudgetContext` — no flag needed. Each `OpRecord` carries `op_name`, `subscripts`, `shapes`, `flop_cost`, `cumulative`, `namespace`, per-op backend/overhead timings, and `resolved_dtype` (which dtype priced the op — the fastest way to find a stray float64). (`whest run --profile` is a separate, unrelated flag: it controls whether the *report* renders the breakdowns.) |
| `summary()` | method | Pretty-printed summary for the current context. |
| `summary_dict(...)` | method | Same data as a `dict` (machine-readable). |
| `deduct(op_name, *, flop_cost, shapes, dtypes, …)` | method | Manually attribute FLOPs. The signature changed in flopscope 0.9: name the op and pass `flop_cost` plus the operand `dtypes` — the dtype rate is applied on top (e.g. `ctx.deduct("my_op", flop_cost=10, subscripts=None, shapes=(), dtypes=(fnp.float64,))` charges 20). Pass `dtypes=()` for a dtype-neutral charge; `dtypes=None` raises. |

```python
with flops.BudgetContext(flop_budget=10_000_000) as ctx:
    # ... your computations ...
    print(ctx.flops_used, "/", ctx.flop_budget)   # quick check
    print(ctx.flops_remaining)
    print(ctx.summary())                          # rich per-op breakdown
    flops.budget_summary()                        # process/session-wide (prints itself)
```

`ctx.summary()` *returns* a string; `flops.budget_summary()` *prints* a rich
panel and returns `None`. Wrap the first in `print()`, not the second. Use
`ctx.summary_dict()` / `flops.budget_summary_dict()` when you want the same data
as a dict.

The session-wide `flops.budget_summary()` and `flops.budget_summary_dict()`
aggregate across every context entered in the current Python process, which
helps when you're profiling a multi-stage pipeline.

Both summaries also include four timing fields that satisfy this strict
timing identity:

```text
wall_time_s = flopscope_backend_time_s + flopscope_overhead_time_s + residual_wall_time_s
```

- `wall_time_s`: total elapsed time in the context
- `flopscope_backend_time_s`: time spent inside counted flopscope numpy kernels (the participant's actual numpy compute)
- `flopscope_overhead_time_s`: time spent inside flopscope's own dispatch (wrapper preambles, FLOP bookkeeping, namespace push/pop) — framework cost, not participant cost
- `residual_wall_time_s`: participant Python (loops, control flow), GC, and Python-callback op time; as of flopscope 0.7.0, data-movement NumPy ops (concatenate, stack, tile, repeat, take, pad, …) count as `flopscope_backend_time_s`, not residual

## WhestBench-specific limits

Flopscope's `BudgetContext` measures `wall_time_s`, `flopscope_backend_time_s`,
`flopscope_overhead_time_s`, and `residual_wall_time_s`. It also accepts
`wall_time_limit_s`, which it checks while counted flopscope operations run.

WhestBench exposes some of those concepts as run-level CLI knobs:

- `--wall-time-limit`: passed through to the estimator's `BudgetContext`.
  The grader sets it to **120 s per MLP**.
- `--residual-wall-time-limit`: enforced by WhestBench after `predict()` returns,
  using the reported `residual_wall_time_s`. Because `residual_wall_time_s`
  excludes flopscope backend and dispatch time, this gate measures only your
  Python and uninstrumented work, not numpy backend execution or the
  framework's bookkeeping tax. The grader sets it to **400 ms per MLP**.

So if you see `time_exhausted`, that came from Flopscope's `wall_time_limit_s`.
If you see `residual_wall_time_exhausted`, that came from WhestBench scoring
logic comparing Flopscope's measured `residual_wall_time_s` with the configured
`--residual-wall-time-limit`.

Both are hard caps: crossing either one zeroes that MLP's predictions. Neither
is priced. There is no rate at which you can buy extra residual time by
accepting a worse multiplier. The 400 ms allowance exists so your Python can
move results between flopscope calls; performing meaningful computation in it
is a rules violation and grounds for disqualification. See
[Estimator Contract: Phase 2 limits](estimator-contract.md#phase-2-limits).

## Common Gotchas

**numpy arrays still count FLOPs.** Since `fnp.ndarray` is backed by numpy, a raw numpy array passed to flopscope operations will still be tracked. Use `fnp.array()` or `fnp.asarray()` to convert explicitly.

**Pythonic operators are tracked.** `x @ w` counts the same FLOPs as `fnp.matmul(x, w)`. Use whichever reads better.

**dtype now matters for FLOPs too.** Since flopscope 0.9, float64 operations
bill 2× float32 (`dtype_rate`). NumPy promotion decides the billing dtype, so
one float64 operand upgrades the whole expression. Keep estimator state in
float32 unless you need the precision, and note that `fnp.zeros(...)` defaults
to float64. flopscope admits only numeric dtypes: `dtype.kind in "biufc"` (bool,
signed/unsigned integer, float, complex). Anything else (object, str, bytes,
datetime64, timedelta64, structured/void) raises `UnsupportedDtypeError` before
any FLOPs are charged, whether the offending dtype arrives as an operand, an
explicit `dtype=`, a fill value, a distribution parameter, or an `out=`
destination. **Import it from `flopscope.errors`, not from `flopscope`.** It
subclasses `TypeError`, so `except TypeError` also catches it. In a graded run
this surfaces as a per-MLP failure, so a stray `fnp.array(['a','b'])` zeroes
that MLP.

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

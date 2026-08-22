# `local_engine` API

> [← Documentation](../README.md)

`local_engine.py` is a pedagogical re-implementation of whestbench's MLP factory and Monte-Carlo simulator in raw `flopscope` code. You can read the whole file in 5 minutes — that's the point.

> **Why pedagogical?** Stage 1 is about understanding the math. Re-implementing in raw flopscope means there's no library magic between you and the forward pass.

## `build_mlp(width, depth, seed=0) -> MLP`

Returns a square MLP with He-initialized weights: `N(0, 2/width)` per element. Deterministic given `seed`.

```python
from local_engine import build_mlp
mlp = build_mlp(width=1024, depth=16, seed=0)  # Phase 2 competition shape; Phase 1 was 256x32
```

Constraints: `width >= 1`, `depth >= 1`. Otherwise raises `ValueError`.

## `monte_carlo_layer_means(mlp, n_samples, seed=0) -> fnp.ndarray`

Forwards `n_samples` independent N(0, 1) inputs through `mlp.weights` and returns the per-layer mean post-activation. Shape: `(mlp.depth, mlp.width)`.

```python
from local_engine import monte_carlo_layer_means
truth = monte_carlo_layer_means(mlp, n_samples=10_000, seed=0)
```

## `compare_against_monte_carlo(estimator, mlp, ...) -> None`

```python
compare_against_monte_carlo(
    estimator,
    mlp,
    sample_counts=(10, 100, 1_000, 10_000, 100_000),
    estimator_budget=2**41,
    sampling_budget=int(5e12),
    seed=0,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `sample_counts` | `(10, 100, 1_000, 10_000, 100_000)` | The full five-row sweep takes about 3 s at the Phase 2 shape. |
| `estimator_budget` | `2**41` = `2,199,023,255,552` | The Phase 2 per-MLP budget, so Stage 1 rehearses against the real cap. |
| `sampling_budget` | `int(5e12)` | Applied **per row** — each sample count runs in its own `BudgetContext` — so it has to clear the largest single row (3,363,737,665,536 FLOPs at `n_samples=100_000`), not the sum of the sweep. |
| `seed` | `0` | Seeds the Monte-Carlo input draws. |

Runs your estimator once, then sweeps Monte Carlo at each `sample_counts` value, printing a convergence table:

```
 n_samples | sampling_flops | estimator_flops |        MSE
-----------------------------------------------------------
        10 |    336,439,296 |         131,072 |   <your MSE>
       ...
```

The FLOP columns are the measured costs at `width=1024, depth=16` with
[`examples/01_random.py`](../../examples/01_random.py) as the estimator; the
MSE column is whatever your own estimator achieves. `estimator_flops` is
constant down the column, because your estimator runs once, before the sweep.

**Friendly preflight:** before the MC sweep, the function checks that `estimator.predict(mlp, budget)` returns a `flopscope.numpy.ndarray` of shape `(depth, width)`. On failure it prints a one-line diagnostic pointing at [estimator-contract.md](estimator-contract.md) and raises `SystemExit(2)` — no numpy traceback. It does **not** check dtype: an integer-dtype return of the right shape passes here *and* passes `whest run`, so cast to `fnp.float32` yourself.

Returns `None` — this is a print helper for stage-1 dev loops.

## Monte-Carlo cost at the Phase 2 shape

Measured at `width=1024, depth=16` under the flopscope version this kit pins (`>=0.12.0,<0.13.0`). These totals are unchanged from 0.11.0: none of the ~40 operations 0.12.0 repriced appear in this path, which is what `tests/test_flopscope_cost_docs.py` pins:

| `n_samples` | sampling FLOPs | wall time |
|---|---|---|
| 10 | 336,439,296 | 0.01 s |
| 100 | 3,363,803,136 | 0.01 s |
| 1,000 | 33,637,441,536 | 0.03 s |
| 10,000 | 336,373,825,536 | 0.25 s |
| 100,000 | 3,363,737,665,536 | 2.68 s |

One forward pass costs ≈ 33,637,376 FLOPs, so a per-MLP budget of `2**41`
buys roughly 65,374 Monte-Carlo passes. The 100,000-sample sweep in the
bottom row therefore costs more than an entire per-MLP budget: brute-force
sampling is a ground-truth instrument, not a viable estimator.

## Parity with whestbench

`local_engine.build_mlp` is statistically equivalent to `whestbench.sample_mlp`. CI asserts this (`tests/test_local_engine_parity.py`).

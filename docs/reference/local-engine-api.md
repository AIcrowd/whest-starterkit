# `local_engine` API

`local_engine.py` is a pedagogical re-implementation of whestbench's MLP factory and Monte-Carlo simulator in raw `whest` code. You can read the whole file in 5 minutes — that's the point.

> **Why pedagogical?** Stage 1 is about understanding the math. Re-implementing in raw whest means there's no library magic between you and the forward pass.

## `build_mlp(width, depth, seed=0) -> MLP`

Returns a square MLP with He-initialized weights — `N(0, 2/width)` per element. Deterministic given `seed`.

```python
from local_engine import build_mlp
mlp = build_mlp(width=32, depth=6, seed=0)
```

Constraints: `width >= 1`, `depth >= 1`. Otherwise raises `ValueError`.

## `monte_carlo_layer_means(mlp, n_samples, seed=0) -> we.ndarray`

Forwards `n_samples` independent N(0, 1) inputs through `mlp.weights` and returns the per-layer mean post-activation. Shape: `(mlp.depth, mlp.width)`.

```python
from local_engine import monte_carlo_layer_means
truth = monte_carlo_layer_means(mlp, n_samples=10_000, seed=0)
```

## `compare_against_monte_carlo(estimator, mlp, sample_counts=..., ...) -> None`

Runs your estimator once, then sweeps Monte Carlo at each `sample_counts` value, printing a convergence table:

```
 n_samples | sampling_flops | estimator_flops |        MSE
-----------------------------------------------------------
        10 |       614,400 |         24,576 |   0.013214
       ...
```

**Friendly preflight:** before the MC sweep, the function checks that `estimator.predict(mlp, budget)` returns the right shape and dtype. On failure, prints a one-line diagnostic pointing at [estimator-contract.md](estimator-contract.md) and exits cleanly (no numpy traceback).

Returns `None` — this is a print helper for stage-1 dev loops.

## Parity with whestbench

`local_engine.build_mlp` is statistically equivalent to `whestbench.sample_mlp`. CI asserts this (`tests/test_local_engine_parity.py`).

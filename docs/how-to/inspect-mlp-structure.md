# Inspect and Traverse MLP Structure

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page when implementing estimator logic that depends on MLP topology or layer weights.

## 📌 TL;DR

- `MLP.width`: number of neurons per layer.
- `MLP.depth`: number of layers.
- `MLP.weights`: ordered list of weight matrices, each shape `(width, width)`.
- `MLP.seed`: the grader's per-MLP seed, and the only correct source of randomness in `predict()`.

At the Phase 2 competition shape that is `width = 1024`, `depth = 16`: sixteen
1024×1024 matrices, and a `(16, 1024)` prediction. Read both off `mlp` rather
than hard-coding them. The shape changed from Phase 1's 256×32 to this one.

## 🚀 Do this now

Use this traversal pattern inside `predict`:

```python
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp

from whestbench import BaseEstimator, MLP


class Estimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        # float32 throughout. flopscope bills float64 at 2x, and `fnp.zeros`
        # defaults to float64 — NumPy promotion then spreads it through the
        # whole loop, matmuls included. Dropping these four dtype pins costs
        # 153,913,344 FLOPs instead of 86,639,616, with no warning to tell you.
        mu = fnp.zeros(mlp.width, dtype=fnp.float32)
        var = fnp.ones(mlp.width, dtype=fnp.float32)

        rows = []
        for w in mlp.weights:
            # w has shape (width, width)
            mu_pre = w.T @ mu
            var_pre = (w * w).T @ var
            var_pre = fnp.maximum(var_pre, 1e-12)
            sigma_pre = fnp.sqrt(var_pre)

            alpha = mu_pre / sigma_pre

            # Compute phi(alpha) and Phi(alpha) for the ReLU expectation.
            # `flops.stats.norm.*` promotes float32 input to float64 to match
            # scipy.stats, so cast straight back or the promotion re-infects
            # the rest of the loop at the 2x rate.
            phi_alpha = flops.stats.norm.pdf(alpha).astype(fnp.float32)
            Phi_alpha = flops.stats.norm.cdf(alpha).astype(fnp.float32)

            # E[ReLU(pre)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # E[z^2] = (mu_pre^2 + var_pre) * Phi(alpha) + mu_pre * sigma_pre * phi(alpha)
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            # Var[ReLU] = E[z^2] - E[z]^2
            var = fnp.maximum(ez2 - mu * mu, 0.0)

            rows.append(mu)

        return fnp.stack(rows, axis=0)
```

## MLP fields

| Object | Field | Meaning | Shape / Type |
|---|---|---|---|
| `MLP` | `width` | Number of neurons per layer | `int` |
| `MLP` | `depth` | Number of weight matrices (layers) | `int` |
| `MLP` | `weights` | Ordered weight matrices from layer 0 to `depth-1` | `list[fnp.ndarray]`, each `(width, width)` **float32** |
| `MLP` | `seed` | Per-MLP grader-supplied seed. Seed predict-time randomness from it (`fnp.random.default_rng(mlp.seed)`) or your submission will not reproduce under regrade. `0` when unavailable. | `int` |
| `MLP` | `name` | Human-readable per-MLP slug, e.g. `kathleen-mueller`. Stable across runs; useful in log lines. Empty string outside an evaluator path. | `str` |

Each weight matrix has shape `(width, width)`, 1024×1024 in Phase 2. The pre-activation for layer `l` is computed as `W_l^T @ x` where `x` is the post-activation output of the previous layer.

The traversal above is the algorithm in [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py), which costs **86,639,616 FLOPs** per `predict()` at this shape (0.004% of the `2^41` per-MLP budget), with the two matrix–vector products per layer accounting for 77.42% of it. Those are `O(width²)` each. Propagating a full covariance instead turns them into matrix–matrix products at `O(width³)`, which at width 1024 is what makes [`examples/03_covariance_propagation.py`](../../examples/03_covariance_propagation.py) 597× more expensive than this loop.

## ReLU activation

Each layer applies a ReLU activation: `y = max(0, W^T @ x)`. For mean estimation under Gaussian approximations:

```
E[ReLU(z)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
```

where `alpha = mu_pre / sigma_pre`, `Phi` is the normal CDF, and `phi` is the normal PDF.

## ✅ Expected outcome

You can inspect any layer's weight matrix and implement layer-wise update rules without guessing object structure.

## Notes

- Weight matrices are dense: each `(width, width)` matrix encodes all neuron connections at that layer.
- Estimators must return a `(mlp.depth, mlp.width)` array, `(16, 1024)` in Phase 2.
- Width 1024 also makes memory a real constraint: the solution process gets 8 GB, and one `(width, width)` covariance now holds 16× the elements it did at width 256. Count how many of them you keep alive at once.

## ➡️ Next step

- [Write an Estimator](./write-an-estimator.md)
- [Estimator Contract](../reference/estimator-contract.md)
- [Problem Setup](../concepts/problem-setup.md)

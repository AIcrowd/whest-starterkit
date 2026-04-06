# Inspect and Traverse MLP Structure

## When to use this page

Use this page when implementing estimator logic that depends on MLP topology or layer weights.

## TL;DR

- `MLP.width`: number of neurons per layer.
- `MLP.depth`: number of layers.
- `MLP.weights`: ordered list of weight matrices, each shape `(width, width)`.

## Do this now

Use this traversal pattern inside `predict`:

```python
import mechestim as me
import numpy as np
from scipy.special import ndtr


def _norm_pdf(x: me.ndarray) -> me.ndarray:
    """Standard normal PDF: phi(x) = exp(-x^2 / 2) / sqrt(2*pi)."""
    return me.multiply(
        me.exp(me.multiply(-0.5, me.multiply(x, x))),
        1.0 / float(np.sqrt(2.0 * np.pi)),
    )


def _norm_cdf(x: me.ndarray) -> me.ndarray:
    """Standard normal CDF via scipy's ndtr."""
    return me.array(ndtr(np.asarray(x, dtype=np.float64)).astype(np.float32))


def predict(self, mlp: MLP, budget: int) -> me.ndarray:
    mu = me.zeros(mlp.width)
    var = me.ones(mlp.width)

    rows = []
    for w in mlp.weights:
        # w has shape (width, width)
        mu_pre = me.matmul(me.transpose(w), mu)
        var_pre = me.matmul(me.transpose(me.multiply(w, w)), var)
        var_pre = me.maximum(var_pre, 1e-12)
        sigma_pre = me.sqrt(var_pre)

        alpha = me.divide(mu_pre, sigma_pre)

        # Compute phi(alpha) and Phi(alpha) for the ReLU expectation
        phi_alpha = _norm_pdf(alpha)
        Phi_alpha = _norm_cdf(alpha)

        # E[ReLU(pre)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
        mu = me.add(
            me.multiply(mu_pre, Phi_alpha),
            me.multiply(sigma_pre, phi_alpha),
        )

        # E[z^2] = (mu_pre^2 + var_pre) * Phi(alpha) + mu_pre * sigma_pre * phi(alpha)
        ez2 = me.add(
            me.multiply(me.add(me.multiply(mu_pre, mu_pre), var_pre), Phi_alpha),
            me.multiply(me.multiply(mu_pre, sigma_pre), phi_alpha),
        )
        # Var[ReLU] = E[z^2] - E[z]^2
        var = me.maximum(me.subtract(ez2, me.multiply(mu, mu)), 0.0)

        rows.append(mu)

    return me.stack(rows, axis=0)
```

## MLP fields

| Object | Field | Meaning | Shape / Type |
|---|---|---|---|
| `MLP` | `width` | Number of neurons per layer | `int` |
| `MLP` | `depth` | Number of weight matrices (layers) | `int` |
| `MLP` | `weights` | Ordered weight matrices from layer 0 to `depth-1` | `list[NDArray[float32]]` |

Each weight matrix has shape `(width, width)`. The pre-activation for layer `l` is computed as `W_l^T @ x` where `x` is the post-activation output of the previous layer.

## ReLU activation

Each layer applies a ReLU activation: `y = max(0, W^T @ x)`. For mean estimation under Gaussian approximations:

```
E[ReLU(z)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
```

where `alpha = mu_pre / sigma_pre`, `Phi` is the normal CDF, and `phi` is the normal PDF.

## Expected outcome

You can inspect any layer's weight matrix and implement layer-wise update rules without guessing object structure.

## Notes

- Weight matrices are dense: each `(width, width)` matrix encodes all neuron connections at that layer.
- Estimators must return a `(mlp.depth, mlp.width)` array.

## Next step

- [Write an Estimator](./write-an-estimator.md)
- [Estimator Contract](../reference/estimator-contract.md)
- [Problem Setup](../concepts/problem-setup.md)

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
import whest as we

# Abramowitz & Stegun approximation constants
_P = 0.2316419
_A1, _A2, _A3 = 0.319381530, -0.356563782, 1.781477937
_A4, _A5 = -1.821255978, 1.330274429


def _norm_pdf(x: we.ndarray) -> we.ndarray:
    """Standard normal PDF: phi(x) = exp(-x^2 / 2) / sqrt(2*pi)."""
    return we.exp(-0.5 * x * x) / we.sqrt(2.0 * we.pi)


def _norm_cdf(x: we.ndarray) -> we.ndarray:
    """Standard normal CDF using A&S approximation. Accurate to < 7.5e-8."""
    t = 1.0 / (1.0 + _P * we.abs(x))
    poly = ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    pdf = we.exp(-0.5 * x * x) / we.sqrt(2.0 * we.pi)
    cdf = 1.0 - pdf * poly
    return we.where(x >= 0, cdf, 1.0 - cdf)


def predict(self, mlp: MLP, budget: int) -> we.ndarray:
    mu = we.zeros(mlp.width)
    var = we.ones(mlp.width)

    rows = []
    for w in mlp.weights:
        # w has shape (width, width)
        mu_pre = we.matmul(we.transpose(w), mu)
        var_pre = we.matmul(we.transpose(we.multiply(w, w)), var)
        var_pre = we.maximum(var_pre, 1e-12)
        sigma_pre = we.sqrt(var_pre)

        alpha = we.divide(mu_pre, sigma_pre)

        # Compute phi(alpha) and Phi(alpha) for the ReLU expectation
        phi_alpha = _norm_pdf(alpha)
        Phi_alpha = _norm_cdf(alpha)

        # E[ReLU(pre)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
        mu = we.add(
            we.multiply(mu_pre, Phi_alpha),
            we.multiply(sigma_pre, phi_alpha),
        )

        # E[z^2] = (mu_pre^2 + var_pre) * Phi(alpha) + mu_pre * sigma_pre * phi(alpha)
        ez2 = we.add(
            we.multiply(we.add(we.multiply(mu_pre, mu_pre), var_pre), Phi_alpha),
            we.multiply(we.multiply(mu_pre, sigma_pre), phi_alpha),
        )
        # Var[ReLU] = E[z^2] - E[z]^2
        var = we.maximum(we.subtract(ez2, we.multiply(mu, mu)), 0.0)

        rows.append(mu)

    return we.stack(rows, axis=0)
```

## MLP fields

| Object | Field | Meaning | Shape / Type |
|---|---|---|---|
| `MLP` | `width` | Number of neurons per layer | `int` |
| `MLP` | `depth` | Number of weight matrices (layers) | `int` |
| `MLP` | `weights` | Ordered weight matrices from layer 0 to `depth-1` | `list[we.ndarray]` |

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

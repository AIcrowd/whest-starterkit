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
def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
    mu = np.zeros(mlp.width, dtype=np.float64)
    var = np.ones(mlp.width, dtype=np.float64)

    rows = []
    for w in mlp.weights:
        W = w.astype(np.float64)
        mu_pre = W.T @ mu
        var_pre = (W ** 2).T @ var
        var_pre = np.maximum(var_pre, 1e-12)
        sigma_pre = np.sqrt(var_pre)

        alpha = mu_pre / sigma_pre
        phi_alpha = norm.pdf(alpha)
        Phi_alpha = norm.cdf(alpha)

        mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha
        ez2 = (mu_pre ** 2 + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
        var = np.maximum(ez2 - mu ** 2, 0.0)

        rows.append(mu.astype(np.float32))

    return np.stack(rows, axis=0)
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

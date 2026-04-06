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
        # (see examples/estimators/mean_propagation.py for full helpers)
        # ...

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

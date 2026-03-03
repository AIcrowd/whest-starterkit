# Inspect and Traverse Circuit Structure

## When to use this page

Use this page when implementing estimator logic that depends on circuit topology or layer coefficients.

## TL;DR

- `Circuit.n`: width (number of wires).
- `Circuit.d`: depth (number of layers).
- `Circuit.gates`: ordered list of `Layer` objects, one per depth.
- Each `Layer` stores vectorized wiring and coefficients for all output wires.

## Do this now

Use this traversal pattern inside `predict`:

```python
for depth_idx, layer in enumerate(circuit.gates):
    # layer.first[i] and layer.second[i] are input-wire indices
    # used to compute output wire i at this depth.
    first = x_mean[layer.first]
    second = x_mean[layer.second]
    x_mean = (
        layer.const
        + layer.first_coeff * first
        + layer.second_coeff * second
        + layer.product_coeff * first * second
    ).astype(np.float32)
    yield x_mean
```

## Circuit and Layer fields

| Object | Field | Meaning | Shape / Type |
|---|---|---|---|
| `Circuit` | `n` | Number of wires per layer | `int` |
| `Circuit` | `d` | Number of transition layers | `int` |
| `Circuit` | `gates` | Ordered layers from depth 0 to `d-1` | `list[Layer]` |
| `Layer` | `first` | Input-wire index for first parent of each output wire | `(n,) int32` |
| `Layer` | `second` | Input-wire index for second parent of each output wire | `(n,) int32` |
| `Layer` | `first_coeff` | Linear coefficient on first parent | `(n,) float32` |
| `Layer` | `second_coeff` | Linear coefficient on second parent | `(n,) float32` |
| `Layer` | `const` | Constant term per output wire | `(n,) float32` |
| `Layer` | `product_coeff` | Bilinear coefficient on parent product | `(n,) float32` |

## Vectorized representation you can precompute

`Circuit` provides a direct packer:

```python
packed = circuit.to_vectorized()
# packed.first_idx: (d, n)
# packed.second_idx: (d, n)
# packed.coeff: (d, n, 4) where channels are:
# [const, first_coeff, second_coeff, product_coeff]
```

Interpretation:

- `packed.first_idx[depth, i]`: first parent wire index for output wire `i`.
- `packed.second_idx[depth, i]`: second parent wire index for output wire `i`.

At each depth, define per-wire features and take a row-wise dot product:

```python
for depth_idx in range(circuit.d):
    current_wire_values = x_mean  # shape: (n,), values entering this depth

    parent_a_values = current_wire_values[packed.first_idx[depth_idx]]   # shape: (n,)
    parent_b_values = current_wire_values[packed.second_idx[depth_idx]]  # shape: (n,)

    wire_features = np.stack(
        [
            np.ones_like(parent_a_values),          # bias term
            parent_a_values,                        # first parent wire value
            parent_b_values,                        # second parent wire value
            parent_a_values * parent_b_values,      # interaction term
        ],
        axis=-1,                                    # shape: (n, 4)
    )

    gate_params = packed.coeff[depth_idx]
    # shape: (n, 4)
    # meaning: per-output-wire parameters
    # [const, first_coeff, second_coeff, product_coeff]

    next_wire_values = np.einsum("nk,nk->n", wire_features, gate_params).astype(np.float32)
    # shape: (n,)
    # meaning: predicted expected wire values after this depth layer

    x_mean = next_wire_values
    yield next_wire_values
```

Equivalent form without `einsum`:

```python
next_wire_values = np.sum(wire_features * gate_params, axis=-1).astype(np.float32)
```

For batched states `x` with shape `(batch, n)`, gather parent wires along axis 1:

```python
parent_a_values = x[:, packed.first_idx[depth_idx]]
parent_b_values = x[:, packed.second_idx[depth_idx]]
```

## ✅ Expected outcome

You can inspect any depth layer, read its parent wiring, and implement depth-wise update rules without guessing object structure.

## 📌 Notes

- Layers are vectorized: one `Layer` encodes all output wires at that depth.
- `first` and `second` are index arrays into the previous depth state.
- Estimators must still yield exactly one `(circuit.n,)` row per layer.

## ➡️ Next step

- [Write an Estimator](./write-an-estimator.md)
- [Estimator Contract](../reference/estimator-contract.md)
- [Problem Setup](../concepts/problem-setup.md)

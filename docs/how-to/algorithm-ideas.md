# Algorithm Ideas

This page surveys estimation strategies for the Network Estimation Challenge. Each approach trades accuracy against FLOP cost differently.

## Monte Carlo sampling

Generate random inputs, propagate them through the MLP, average the outputs.

```python
import mechestim as me

def predict_sampling(mlp, budget):
    width = mlp.width
    n_samples = budget // (mlp.depth * width * width * 2)  # rough FLOP estimate per sample
    n_samples = max(n_samples, 1)
    x = me.array(me.random.default_rng().standard_normal((n_samples, width)).astype(me.float32))
    rows = []
    for w in mlp.weights:
        x = me.maximum(me.matmul(x, w), 0.0)
        rows.append(me.mean(x, axis=0))
    return me.stack(rows, axis=0)
```

**FLOP cost:** O(samples x depth x width^2). For width=100, depth=16, one sample costs ~320K FLOPs. 100 samples costs ~32M FLOPs.

**When to use:** As a baseline or sanity check. Accuracy improves as 1/sqrt(samples) — slow convergence.

## Mean propagation (diagonal variance)

Track per-neuron means and variances through each layer using the ReLU expectation formula. Assumes neurons are independent (diagonal covariance).

**FLOP cost:** O(depth x width^2) — one matrix-vector multiply per layer. For width=100, depth=16: ~3.2M FLOPs.

**When to use:** Default choice for most budgets. Fast and reasonably accurate for shallow-to-medium networks.

**Example:** [`examples/estimators/mean_propagation.py`](../../examples/estimators/mean_propagation.py)

## Covariance propagation (full matrix)

Track the full covariance matrix between neurons. More accurate because it captures correlations that diagonal methods ignore.

**FLOP cost:** O(depth x width^3) — matrix-matrix multiply per layer. For width=100, depth=16: ~3.2B FLOPs. Much more expensive.

**When to use:** When the FLOP budget is large relative to width^2, and accuracy matters more than speed. Best for narrow networks or shallow depths.

**Example:** [`examples/estimators/covariance_propagation.py`](../../examples/estimators/covariance_propagation.py)

## Hybrid / budget-aware routing

Switch between cheap and expensive strategies based on the available FLOP budget:

```python
def predict(self, mlp, budget):
    if budget >= 30 * mlp.width * mlp.width:
        return self._covariance_path(mlp)
    return self._mean_path(mlp)
```

**When to use:** When you want a single estimator that adapts to different budget levels.

**Example:** [`examples/estimators/combined_estimator.py`](../../examples/estimators/combined_estimator.py)

## Open directions

These are approaches the organizers think are promising but haven't been tried in this challenge:

- **Low-rank covariance approximation:** Track a rank-k approximation of the covariance matrix. Cost O(depth x width^2 x k) — between diagonal and full.
- **Layer-adaptive methods:** Use full covariance for the first few layers (where correlations build up), then switch to diagonal for deeper layers.
- **Spectral approaches:** Analyze the weight matrices' singular values to predict how information flows.
- **Importance sampling:** Sample inputs that are more informative for estimating deep-layer means.
- **Moment matching beyond second order:** Track skewness or kurtosis to improve ReLU approximations.

## Next step

- [Manage Your FLOP Budget](./manage-flop-budget.md)
- [Performance Tips](./performance-tips.md)
- [From Problem to Code](../getting-started/from-problem-to-code.md)

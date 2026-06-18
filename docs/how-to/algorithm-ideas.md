# Algorithm Ideas

> [← Documentation](../README.md)

This page surveys estimation strategies for the ARC Whitebox Estimation Challenge. Each approach trades accuracy against FLOP cost differently.

## Monte Carlo sampling

Generate random inputs, propagate them through the MLP, average the outputs.

```python
import flopscope.numpy as fnp

def predict_sampling(mlp, budget):
    width = mlp.width
    n_samples = budget // (mlp.depth * width * width)  # rough FLOP estimate per sample
    n_samples = max(n_samples, 1)
    x = fnp.array(fnp.random.default_rng().standard_normal((n_samples, width)).astype(fnp.float32))
    rows = []
    for w in mlp.weights:
        x = fnp.maximum(fnp.matmul(x, w), 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)
```

**FLOP cost:** O(samples x depth x width^2). For width=256, depth=32, one sample costs ~4M FLOPs. 100 samples costs ~420M FLOPs. (The warmup 256×8 round was ~4× cheaper: ~1M/sample, ~105M for 100.)

**When to use:** As a baseline or sanity check. Accuracy improves as 1/sqrt(samples) — slow convergence.

## Mean propagation (diagonal variance)

Track per-neuron means and variances through each layer using the ReLU expectation formula. Assumes neurons are independent (diagonal covariance).

**FLOP cost:** O(depth x width^2) — matrix-vector products per layer. For width=256, depth=32: ~11M FLOPs (the warmup 256×8 round was ~2.7M; see the worked walkthrough in [Manage Your FLOP Budget](./manage-flop-budget.md)).

**When to use:** Default choice for most budgets. Fast and reasonably accurate for shallow-to-medium networks.

**Example:** [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py)

## Covariance propagation (full matrix)

Track the full covariance matrix between neurons. More accurate because it captures correlations that diagonal methods ignore.

**FLOP cost:** O(depth x width^3) — matrix-matrix products per layer. For width=256, depth=32: ~1.6B FLOPs (the warmup 256×8 round was ~404M). Much more expensive.

**When to use:** When the FLOP budget is large relative to width^2, and accuracy matters more than speed. Best for narrow networks or shallow depths.

**Example:** [`examples/03_covariance_propagation.py`](../../examples/03_covariance_propagation.py)

## Fixed single-strategy design

The grader’s budget is fixed for a given run. For stable behavior, this
starter kit now uses single-strategy baselines (mean propagation and full
covariance propagation), then encourages tuning one strategy for that fixed
budget envelope.

## Open directions

These are approaches the organizers think are promising but haven't been
tried in this challenge. Each entry: one-line intuition, complexity, and
when it's likely to pay off.

**Low-rank covariance.** Carry a rank-`k` factor `U` of shape `(width, k)`
so `cov ≈ U Uᵀ` instead of the full `(width, width)` matrix. Cost
`O(depth · width² · k)` per layer — between diagonal (k=1) and full
(k=width). Try when your full covariance fits accuracy-wise but burns
the budget; pick `k` from the spectrum of the early layers' covariance.
Reference: any low-rank Kalman filter / Ensemble Kalman filter intro.

**Layer-adaptive routing.** Use full covariance for the first few layers
(where correlations *build*) and switch to diagonal once the
joint distribution looks roughly factored. Cost is the integral of the
per-layer choice. Look at per-layer `all_layers_mse` from a
covariance-only baseline — the layer where the curve plateaus is your
crossover point. Layer-adaptive switching is one possible per-layer optimization
on top of a full-covariance implementation.

**Spectral / weight-statistics methods.** Compute singular values of
each `W` once in `setup()` (off-budget); use them to predict per-layer
gain and variance growth analytically without propagating any
distribution through the layers. Cost `O(depth · width³)` in setup,
near-zero per `predict()` call. Mostly an academic angle today —
sensitive to depth and to the He-init scaling, but a candidate for
extreme-budget regimes. References: Pennington & Worah (2017), Saxe et
al. (2014).

**Importance sampling.** Bias the input distribution toward regions
where deep-layer activations have high variance, then re-weight
(`sum w_i · ReLU(...) / sum w_i`). Cost `O(samples · depth · width²)`
plus the cost of designing the proposal. Try when standard MC plateaus
above `~1/√samples` for a particular MLP — usually because most random
inputs activate few neurons. Reference: any bridge-sampling /
importance-sampling tutorial.

**Higher-order moments.** Track skewness (third moment) or kurtosis
(fourth moment) per neuron in addition to mean and variance. Cost
`O(depth · width^k)` for the `k`-th moment. The ReLU expectation formula
above assumes Gaussian pre-activations; tracking a third moment lets you
correct for the asymmetry that builds up in deeper networks. Reference:
mean-field analyses such as Schoenholz et al. (2017).

## ➡️ Next step

- [Manage Your FLOP Budget](./manage-flop-budget.md)
- [Performance Tips](./performance-tips.md)
- [Stage 1: Iterate Locally](../getting-started/stage-1-standalone.md)

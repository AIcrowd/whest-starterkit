# Algorithm Ideas

> [← Documentation](../README.md)

This page surveys estimation strategies for the ARC Whitebox Estimation Challenge. Each approach trades accuracy against FLOP cost differently.

## Monte Carlo sampling

Generate random inputs, propagate them through the MLP, average the outputs.

```python
import flopscope.numpy as fnp

def predict_sampling(mlp, budget):
    width = mlp.width
    # Rough per-sample cost: 2 * depth * width^2 for the matmuls. The ReLU, the
    # per-layer mean and the RNG draw are on top, so leave headroom.
    per_sample = 2 * mlp.depth * width * width
    n_samples = max(1, int(0.9 * budget) // per_sample)
    rng = fnp.random.default_rng(mlp.seed)  # per-MLP seed, per the estimator contract
    x = rng.standard_normal((n_samples, width), dtype=fnp.float32)
    rows = []
    for w in mlp.weights:
        x = fnp.maximum(fnp.matmul(x, w), 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)
```

**FLOP cost:** O(samples x depth x width^2). At the competition shape (`width=1024, depth=16`) this snippet costs 33,603,584 FLOPs for a single sample and 33,587,200 for each additional one, so 100 samples cost 3,358,736,384 and the full per-MLP budget of 2,199,023,255,552 buys roughly 65,000 samples (the snippet's own `0.9 * budget` headroom caps it near 59,000). The kit's ground-truth sampler, `local_engine.monte_carlo_layer_means`, bills slightly more (33,637,376 per additional sample) because it accumulates each layer mean in float64; that is the figure quoted in [Local Engine API](../reference/local-engine-api.md).

**Memory:** the solution process is capped at 8 GB, and the FLOP budget is not the only ceiling. A 65,000-sample batch is ~270 MB per `(samples, width)` float32 activation tensor and the layer loop holds two at a time. Comfortable here, but chunk the batch rather than growing it if you push the sample count further.

**When to use:** As a baseline or sanity check. Accuracy improves as 1/sqrt(samples), so convergence is slow.

## Mean propagation (diagonal variance)

Track per-neuron means and variances through each layer using the ReLU expectation formula. Assumes neurons are independent (diagonal covariance).

**FLOP cost:** O(depth x width^2), from matrix-vector products per layer. At `width=1024, depth=16`: 86,639,616 FLOPs, about 0.004% of the per-MLP budget (see the worked walkthrough in [Manage Your FLOP Budget](./manage-flop-budget.md#worked-walkthrough-mean-propagation-line-by-line)).

**When to use:** The cheapest reasonable starting point, and the baseline to beat. Fast and reasonably accurate, but it leaves almost the entire budget unspent. Treat it as a floor rather than a destination.

**Example:** [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py)

## Covariance propagation (full matrix)

Track the full covariance matrix between neurons. More accurate because it captures correlations that diagonal methods ignore.

**FLOP cost:** O(depth x width^3), from matrix-matrix products per layer. At `width=1024, depth=16`: 51,709,240,799 FLOPs, or 2.351% of the per-MLP budget, including the per-layer `as_symmetric()` re-validation that keeps the big einsum on flopscope's symmetric rate. That is 597× mean propagation, and still under 3% of the budget.

**When to use:** Whenever the extra accuracy is worth the implementation effort. Affordability is not the constraint at this shape: 2.351% of the budget sits below the score's `max(0.1, C_m / B_m)` multiplier floor, so those extra FLOPs cost you nothing in the compute term.

**Example:** [`examples/03_covariance_propagation.py`](../../examples/03_covariance_propagation.py)

## Fixed single-strategy design

The grader’s budget is fixed: `B_m = 2**41 = 2,199,023,255,552` FLOPs per MLP,
at a fixed shape of `width=1024, depth=16`. For stable behavior, this starter
kit uses single-strategy baselines (mean propagation and full covariance
propagation), then encourages tuning one strategy for that fixed budget
envelope. There is no need for runtime strategy selection: you already know
the budget and the shape before you write a line of code.

## Open directions

These are approaches the organizers think are promising but haven't been
tried in this challenge. Each entry: one-line intuition, complexity, and
when it's likely to pay off.

**Low-rank covariance.** Carry a rank-`k` factor `U` of shape `(width, k)`
so `cov ≈ U Uᵀ` instead of the full `(width, width)` matrix. Cost
`O(depth · width² · k)` overall, between diagonal (k=1) and full
(k=width). At `width=1024` this is the natural way to spend budget on
correlations you cannot afford to model exactly; pick `k` from the
spectrum of the early layers' covariance.
Reference: any low-rank Kalman filter / Ensemble Kalman filter intro.

**Layer-adaptive routing.** Use full covariance for the layers where
correlations *build* and switch to diagonal once the joint distribution
looks roughly factored. Cost is the integral of the per-layer choice.
Look at per-layer `all_layers_mse` from a covariance-only baseline. The
layer where the curve plateaus is your crossover point. At this
shape full covariance for all 16 layers already fits in 2.4% of the budget,
so routing is about spending your effort where it buys accuracy, not about
making the run affordable.

**Spectral / weight-statistics methods.** Compute singular values of
each `W` once per MLP, then use them to predict per-layer gain and
variance growth analytically without propagating any distribution
through the layers. This has to happen inside `predict()` and is billed
there: the weights are per-MLP, so they don't exist yet in `setup()`.
And `setup()`'s off-budget status can't be used to dodge the
`O(depth · width³)` factorisation, because `setup()` runs once per worker
process (several times per submission), each run under a 5 s wall-clock
cap, and exceeding it fails the whole submission rather than a single
MLP. What you get instead is a near-zero marginal cost once the spectra
are in hand. Mostly an academic angle today, sensitive to depth and to
the He-init scaling, but a candidate for extreme-budget regimes.
References: Pennington & Worah (2017), Saxe et al. (2014).

**Importance sampling.** Bias the input distribution toward regions
where deep-layer activations have high variance, then re-weight
(`sum w_i · ReLU(...) / sum w_i`). Cost `O(samples · depth · width²)`
plus the cost of designing the proposal. Try when standard MC plateaus
above `~1/√samples` for a particular MLP, usually because most random
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

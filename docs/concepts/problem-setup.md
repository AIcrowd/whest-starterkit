# Problem Setup

## When to use this page

Use this page when you want a better understanding of the technical framing of the problem.

## TL;DR

- Input: one random layered `MLP` and one `flop_budget`.
- Output: one `(n,)` prediction row per depth, for exactly `d` depths.
- Goal: estimate expected neuron values under uniformly random inputs.
- Predictions are real-valued expected neuron states, not probabilities.
- Scoring is pure MSE under an analytical FLOP budget constraint — predictions are zeroed if the budget is exceeded.

## The research question

This challenge targets a foundational question in mechanistic estimation:

> **Can you predict a model's behavior by analyzing its structure, rather than just running it on many inputs?**

The natural baseline for estimating a network's expected output is **sampling**: feed in thousands of random inputs, propagate them through the network, and average the results. Sampling is the ground truth — with enough samples it converges to the exact answer. But it's inefficient: it scales as 1/√k and learns nothing from the network's structure.

**Mechanistic estimation** means predicting network behavior using mathematical properties of the architecture — weight statistics, activation functions, input distributions — instead of running forward passes. It is distinct from mechanistic interpretability (understanding what neurons represent) and from symbolic execution (exact but intractable computation). Because sampling scales so poorly, there is room for structural methods to reach the same accuracy in far less compute. The question is whether such methods can actually beat sampling at this task.

ARC's recent work frames "competing with sampling" as an important and difficult milestone:

- [Competing with sampling](https://www.alignment.org/blog/competing-with-sampling/)
- [AlgZoo: uninterpreted models with fewer than 1,500 parameters](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)

This challenge instantiates that question in random MLPs, where evaluation is explicit, reproducible, and compute-aware.

## What is an MLP?

An MLP is a layered computation graph with fixed **width** `n` (the number of neurons per layer) and **depth** `d` (the number of transformation layers).

**Inputs.** The input layer has `n` neurons, each sampled independently from `N(0, 1)` (standard normal). All inputs are uncorrelated with expected value `E[x] = 0`.

**Layers.** Each layer applies a dense matrix multiply followed by ReLU activation:

```
y = ReLU(W.T @ x)
```

where `W` is a `(n, n)` weight matrix initialized with He initialization (`N(0, 2/n)`), and `ReLU(z) = max(z, 0)`. Weight matrices are stored as `(input, output)` following numpy convention, so the forward pass computes `W.T @ x` for a single input vector, or equivalently `x @ W` for batched inputs.

This scaling keeps activations from exploding or vanishing through deep layers. For your estimator, it means the variance entering each layer is predictable — a useful property for analytical approaches.

Every neuron in a layer receives input from **all** neurons in the previous layer (dense connectivity), not a sparse subset.

**Output.** After `d` layers, the network has `n` output neurons. Your job is to estimate the expected value of every neuron after every layer.

## Why depth makes the problem hard

At shallow depth, neurons are nearly independent. A simple approach like **mean propagation** — tracking `E[x]` per neuron and propagating through the ReLU nonlinearity — works reasonably well.

As depth grows, the dense weight matrices create correlations between neurons. ReLU compounds this: it clips negative values, making the output distribution depend on the full joint distribution of its inputs — not just their marginals. These correlations accumulate layer by layer, and the independence assumption that mean propagation relies on breaks down.

This is what makes the problem interesting: you need methods that account for (or at least manage) these growing dependencies — without spending as much compute as sampling would.

## The sampling baseline

The simplest approach is **Monte Carlo sampling**:

1. Draw `k` random input vectors (each neuron independently sampled from `N(0, 1)`).
2. Propagate each input vector through all `d` layers (matmul + ReLU per layer).
3. Average the results per neuron per depth.

This is unbiased and converges as `k → ∞`, but the error decreases slowly (`≈ 1/√k`). The challenge asks: can you reach the same accuracy more efficiently by exploiting the network's structure?

To estimate a width-100, depth-16 MLP to 1% accuracy, sampling needs roughly 10,000 forward passes at ~320K FLOPs each — about 3.2 billion FLOPs total. Mean propagation reaches similar accuracy for ~3.2 million FLOPs. That is a 1000x improvement.

## What the estimator receives

Each evaluation call provides:

- one `MLP` with `n` neurons and `d` layers,
- one integer `flop_budget` — the maximum number of floating-point operations your estimator may use, tracked analytically by whest.

Your estimator must emit exactly `d` vectors, each with shape `(n,)`.

Row `i` is your estimate of expected neuron values after layer `i`.

## Computational model

All FLOP usage is tracked analytically by whest — there is no wall-clock timing. Your estimator imports whest (`import whest as we`) and uses its primitives, which report exact FLOP counts. If the total exceeds `flop_budget`, all predictions for that MLP are zeroed.

## Ground truth

Ground truth is approximated by Monte Carlo simulation over random inputs.
The evaluator computes empirical means by depth and neuron, stored as `ground_truth_samples`.

## ➡️ Next step

- [Scoring Model](./scoring-model.md)
- [Inspect and Traverse MLP Structure](../how-to/inspect-mlp-structure.md)
- [Estimator Contract](../reference/estimator-contract.md)

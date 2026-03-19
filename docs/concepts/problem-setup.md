# Problem Setup

## When to use this page

Use this page when you want a better understanding of the technical framing of the problem.

## TL;DR

- Input: one random layered `MLP` and one `budget`.
- Output: one `(n,)` prediction row per depth, for exactly `d` depths.
- Goal: estimate expected neuron values under uniformly random inputs.
- Predictions are real-valued expected neuron states, not probabilities.
- Scoring rewards accuracy under compute constraints comparable to sampling.

## The research question

This challenge targets a foundational question in mechanistic estimation:

> **Can you predict a model's behavior by analyzing its structure, rather than just running it on many inputs?**

The natural baseline for estimating a network's expected output is **sampling**: feed in thousands of random inputs, propagate them through the network, and average the results. Sampling is the ground truth — with enough samples it converges to the exact answer. But it's inefficient: it scales as 1/√k and learns nothing from the network's structure.

**Mechanistic estimation** takes the opposite approach: instead of brute-force evaluation, analyze the network's topology and layer rules to compute (or approximate) expected neuron values directly. Because sampling scales so poorly, there is room for structural methods to reach the same accuracy in far less compute. The question is whether such methods can actually beat sampling at this task.

ARC's recent work frames "competing with sampling" as an important and difficult milestone:

- [Competing with sampling](https://www.alignment.org/blog/competing-with-sampling/)
- [AlgZoo: uninterpreted models with fewer than 1,500 parameters](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)

This challenge instantiates that question in random MLPs, where evaluation is explicit, reproducible, and compute-aware.

## What is an MLP?

An MLP is a layered computation graph with fixed **width** `n` (the number of neurons) and **depth** `d` (the number of transformation layers).

**Inputs.** At depth 0, every neuron is initialized independently and uniformly at random from `{-1, +1}`. This means every input neuron has expected value `E[x] = 0` and all inputs are uncorrelated.

**Layers.** At each depth, every output neuron reads exactly two input neurons (from the previous layer) and applies an affine-bilinear transformation:

```
y[i] = const[i] + a[i] · x_first[i] + b[i] · x_second[i] + p[i] · x_first[i] · x_second[i]
```

where:
- `first[i]` and `second[i]` select which two neurons from the previous layer feed into output neuron `i`,
- `const`, `a` (first_coeff), `b` (second_coeff), and `p` (product_coeff) are fixed layer parameters.

**Output.** After `d` layers, the network has `n` output neurons. Your job is to estimate the expected value of every neuron after every layer.

## Why depth makes the problem hard

At shallow depth, neurons are nearly independent. A simple approach like **mean propagation** — tracking `E[x]` per neuron and assuming `E[x · y] ≈ E[x] · E[y]` — works reasonably well.

As depth grows, the product term `p · x_first · x_second` creates correlations between neurons. These correlations accumulate layer by layer: neuron A influences neuron B at depth 3, which influences neuron C at depth 5, which feeds back into a descendant of neuron A at depth 8. The independence assumption breaks down, and mean propagation drifts.

This is what makes the problem interesting: you need methods that account for (or at least manage) these growing dependencies — without spending as much compute as sampling would.

## The sampling baseline

The simplest approach is **Monte Carlo sampling**:

1. Draw `k` random input vectors (each neuron independently ±1).
2. Propagate each input vector through all `d` layers.
3. Average the results per neuron per depth.

This is unbiased and converges as `k → ∞`, but the error decreases slowly (`≈ 1/√k`). The challenge asks: can you reach the same accuracy more efficiently by exploiting the network's structure?

## What the estimator receives

Each evaluation call provides:

- one `MLP` with `n` neurons and `d` layers,
- one integer `budget`.

Your estimator must emit exactly `d` vectors, each with shape `(n,)`.

Row `i` is your estimate of expected neuron values after layer `i`.

## Ground truth

Ground truth is approximated by Monte Carlo simulation over random inputs.
The evaluator computes empirical means by depth and neuron.

## ➡️ Next step

- [Scoring Model](./scoring-model.md)
- [Inspect and Traverse MLP Structure](../how-to/inspect-circuit-structure.md)
- [Estimator Contract](../reference/estimator-contract.md)

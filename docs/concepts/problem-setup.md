# Problem Setup

## When to use this page

Use this page when you want a better understanding of the technical framing of the problem.

## TL;DR

- Input: one random layered Boolean-circuit-style `Circuit` and one `budget`.
- Output: one `(n,)` prediction row per depth, for exactly `d` depths.
- Goal: estimate expected wire values under uniformly random inputs.
- Scoring rewards accuracy under compute constraints comparable to sampling.

## The research question

This challenge targets a foundational question in mechanistic estimation:

> **Can you predict a model's behavior by analyzing its structure, rather than just running it on many inputs?**

The natural baseline for estimating a circuit's expected output is **sampling**: feed in thousands of random inputs, propagate them through the circuit, and average the results. Sampling is the ground truth — with enough samples it converges to the exact answer. But it's inefficient: it scales as 1/√k and learns nothing from the circuit's structure.

**Mechanistic estimation** takes the opposite approach: instead of brute-force evaluation, analyze the circuit's wiring and gate rules to compute (or approximate) expected wire values directly. Because sampling scales so poorly, there is room for structural methods to reach the same accuracy in far less compute. The question is whether such methods can actually beat sampling at this task.

ARC's recent work frames "competing with sampling" as an important and difficult milestone:

- [Competing with sampling](https://www.alignment.org/blog/competing-with-sampling/)
- [AlgZoo: uninterpreted models with fewer than 1,500 parameters](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)

This challenge instantiates that question in random Boolean circuits, where evaluation is explicit, reproducible, and compute-aware.

## What is a circuit?

A circuit is a layered computation graph with fixed **width** `n` (the number of wires) and **depth** `d` (the number of transformation layers).

**Inputs.** At depth 0, every wire is initialized independently and uniformly at random from `{-1, +1}`. This means every input wire has expected value `E[x] = 0` and all inputs are uncorrelated.

**Layers.** At each depth, every output wire reads exactly two input wires (from the previous layer) and applies an affine-bilinear gate:

```
y[i] = const[i] + a[i] · x_first[i] + b[i] · x_second[i] + p[i] · x_first[i] · x_second[i]
```

where:
- `first[i]` and `second[i]` select which two wires from the previous layer feed into output wire `i`,
- `const`, `a` (first_coeff), `b` (second_coeff), and `p` (product_coeff) are fixed gate parameters.

**Output.** After `d` layers, the circuit has `n` output wires. Your job is to estimate the expected value of every wire after every layer.

## Why depth makes the problem hard

At shallow depth, wires are nearly independent. A simple approach like **mean propagation** — tracking `E[x]` per wire and assuming `E[x · y] ≈ E[x] · E[y]` — works reasonably well.

As depth grows, the product term `p · x_first · x_second` creates correlations between wires. These correlations accumulate layer by layer: wire A influences wire B at depth 3, which influences wire C at depth 5, which feeds back into a descendant of wire A at depth 8. The independence assumption breaks down, and mean propagation drifts.

This is what makes the problem interesting: you need methods that account for (or at least manage) these growing dependencies — without spending as much compute as sampling would.

## The sampling baseline

The simplest approach is **Monte Carlo sampling**:

1. Draw `k` random input vectors (each wire independently ±1).
2. Propagate each input vector through all `d` layers.
3. Average the results per wire per depth.

This is unbiased and converges as `k → ∞`, but the error decreases slowly (`≈ 1/√k`). The challenge asks: can you reach the same accuracy more efficiently by exploiting the circuit's structure?

## What the estimator receives

Each evaluation call provides:

- one `Circuit` with `n` wires and `d` layers,
- one integer `budget`.

Your estimator must emit exactly `d` vectors, each with shape `(n,)`.

Row `i` is your estimate of expected wire values after layer `i`.

## Ground truth

Ground truth is approximated by Monte Carlo simulation over random inputs.
The evaluator computes empirical means by depth and wire.

## ➡️ Next step

- [Scoring Model](./scoring-model.md)
- [Inspect and Traverse Circuit Structure](../how-to/inspect-circuit-structure.md)
- [Estimator Contract](../reference/estimator-contract.md)

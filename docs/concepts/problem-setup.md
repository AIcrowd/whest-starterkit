# Problem setup

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page to understand the technical framing of the problem.

## 📌 TL;DR

- Input: one random layered `MLP` and one `flop_budget`. This round: width `n = 1024`, depth `d = 16`, `flop_budget = 2**41 = 2,199,023,255,552`.
- Output: one `(n,)` prediction row per depth, for exactly `d` depths, so a `(16, 1024)` array this round.
- Goal: estimate expected neuron values under standard normal `N(0, 1)` inputs.
- Predictions are real-valued expected neuron states, not probabilities.
- Scoring is **budget-adjusted** final-layer MSE: `adjusted_final_layer_score = final_layer_mse × max(0.1, C_m / flop_budget)`, where `C_m = F_m` is the analytical FLOP count and nothing else. If the FLOP budget is exceeded, or the 400 ms residual wall-time cap or the 120 s wall-clock cap trips, predictions are zeroed and the multiplier is forced to 1.0 (no compute discount).

## The research question

This challenge targets an open question in mechanistic estimation:

> **Can you predict a model's behavior by analyzing its structure, rather than running it on many inputs?**

The natural baseline for estimating a network's expected output is **sampling**: feed in thousands of random inputs, propagate them through the network, and average the results. Sampling is the ground truth. With enough samples it converges to the exact answer. But it's inefficient: it scales as 1/√k and makes no use of the network's structure.

**Mechanistic estimation** means predicting network behavior using mathematical properties of the architecture (weight statistics, activation functions, input distributions) instead of running forward passes. It is distinct from mechanistic interpretability (understanding what neurons represent) and from symbolic execution (exact but intractable computation). Because sampling scales so poorly, there is room for structural methods to reach the same accuracy in far less compute.

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

This scaling keeps activations from exploding or vanishing through deep layers. For your estimator, it means the variance entering each layer is predictable.

Every neuron in a layer receives input from **all** neurons in the previous layer (dense connectivity), not a sparse subset.

**Output.** After `d` layers, the network has `n` output neurons. Your job is to estimate the expected value of every neuron after every layer.

## Why depth makes the problem hard

At shallow depth, neurons are nearly independent. An approach such as **mean propagation** — tracking `E[x]` per neuron and propagating through the ReLU nonlinearity — works reasonably well.

As depth grows, the dense weight matrices create correlations between neurons. ReLU compounds this: it clips negative values, making the output distribution depend on the full joint distribution of its inputs, not only their marginals. These correlations accumulate layer by layer, and the independence assumption that mean propagation relies on breaks down.

So you need methods that account for (or at least manage) these growing dependencies without spending as much compute as sampling would.

## The sampling baseline

The most direct approach is **Monte Carlo sampling**:

1. Draw `k` random input vectors (each neuron independently sampled from `N(0, 1)`).
2. Propagate each input vector through all `d` layers (matmul + ReLU per layer).
3. Average the results per neuron per depth.

This is unbiased and converges as `k → ∞`, but the error decreases slowly (`≈ 1/√k`).

Sampling here is expensive, not weak. One forward pass at this shape meters at ~33.6M FLOPs, so the `2**41` per-MLP budget buys roughly **65,000** of them. Measured over the graded 100-MLP `mini` split of `arc-whestbench-public-2026@v2-phase2`: 1,000 passes cost 33,637,441,536 FLOPs (1.5% of budget) and reach a final-layer MSE of `7.8e-05`, about 0.93% RMS relative error; 10,000 passes cost 336,373,825,536 FLOPs (15.3% of budget) and reach `7.6e-06`, about 0.29%. Mean propagation (one O(depth × width²) propagation, no sampling at all) costs 86,639,616 FLOPs, 0.004% of budget, and reaches `2.2e-04`, the accuracy Monte Carlo already has at under 400 passes. So mean propagation is a **~140x compute saving at equal accuracy**, not an accuracy win over sampling.

On the metric you are ranked on, the comparison changes. A plain sampler bottoms out at an `adjusted_final_layer_score` of about `1.2e-06`: measured `1.19e-06` at 6,500 passes (9.9% of budget) and `1.16e-06` at 10,000. Below ~6,500 passes the 0.1 multiplier floor still applies and more samples strictly help; above it the extra multiplier cancels the extra accuracy, so the score goes flat. Mean propagation scores `2.2e-05`, roughly **19x worse** than that sampler. Covariance propagation scores `4.1e-07`, **2.9x better**. Beating sampling is the research milestone this challenge targets, and of the bundled examples only covariance propagation clears it.

## What the estimator receives

Each evaluation call provides:

- one `MLP` with `n` neurons and `d` layers (`n = 1024`, `d = 16` this round),
- one integer `flop_budget`, the maximum number of floating-point operations your estimator may use, tracked analytically by flopscope. This round it is `2**41 = 2,199,023,255,552` per MLP.

Your estimator must emit exactly `d` vectors, each with shape `(n,)`, so a `(16, 1024)` array this round.

Row `i` is your estimate of expected neuron values after layer `i`.

## Computational model

FLOP usage is tracked analytically by flopscope. Your estimator imports flopscope (`import flopscope as flops` and `import flopscope.numpy as fnp`) and uses its primitives, which report exact FLOP counts. The leaderboard ranks on the FLOPs flopscope counted and nothing else: `C_m = F_m`. Wall-clock time is not converted into compute; instead the residual wall-time bucket `R_m` (Python-side work not inside a flopscope kernel) has a hard **400 ms** cap per MLP, alongside a 120 s wall-clock cap. If `C_m > flop_budget`, or either time cap trips, the affected MLP's predictions are zeroed and the budget multiplier is forced to 1.0. `setup()` has a separate hard **5 s** cap, and overrunning that one fails the whole submission with `SETUP_TIMEOUT` rather than a single MLP. See [Scoring Model](./scoring-model.md) for the full formula.

That residual bucket is for plumbing (array marshalling, control flow, bookkeeping), not for computation. The only code you may run is the flopscope client API and the pure-Python standard library; vendored numerical libraries, compiled kernels, FFI, and threads or subprocesses are prohibited, and doing real numerical work outside flopscope's accounting is grounds for disqualification rather than a cost you can choose to pay. Shipping *data* alongside your estimator (weights, lookup tables, precomputed artifacts) remains permitted. The [Estimator Contract](../reference/estimator-contract.md) lists the rules in full.

## Ground truth

Ground truth is approximated by Monte Carlo simulation over standard normal `N(0, 1)` inputs.
The evaluator computes empirical means by depth and neuron, stored as `ground_truth_samples`.

## Further reading

To read further into the literature behind structural estimation,
start with these. Each link resolves on the public web,
and none of them is required to do the challenge.

- [ARC: Competing with sampling](https://www.alignment.org/blog/competing-with-sampling/) — the framing post for this challenge.
- [ARC: AlgZoo — uninterpreted models with fewer than 1,500 parameters](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/) — concrete examples of how structural understanding compresses computation.
- Frey & Hinton, *Variational Learning in Nonlinear Gaussian Belief Networks* (1999) — the classical derivation of the rectified-Gaussian first moment used by the mean-propagation example.
- Schoenholz, Gilmer, Ganguli & Sohl-Dickstein, *Deep Information Propagation* (ICLR 2017, [arXiv:1611.01232](https://arxiv.org/abs/1611.01232)) — mean-field analysis of how moments propagate through deep ReLU networks; explains when/why the Gaussian assumption breaks.
- Pennington & Worah, *Nonlinear random matrix theory for deep learning* (NeurIPS 2017) — spectral approaches to predicting layer-by-layer signal evolution.
- Saxe, McClelland & Ganguli, *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks* (ICLR 2014, [arXiv:1312.6120](https://arxiv.org/abs/1312.6120)) — singular-value-based reasoning that motivates the spectral entry in [algorithm-ideas.md](../how-to/algorithm-ideas.md).

## ➡️ Next step

- [Scoring Model](./scoring-model.md)
- [Inspect and Traverse MLP Structure](../how-to/inspect-mlp-structure.md)
- [Estimator Contract](../reference/estimator-contract.md)

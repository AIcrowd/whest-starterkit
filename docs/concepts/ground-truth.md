# How Ground Truth Is Generated

> [← Documentation](../README.md)

This page explains how the evaluator computes the reference values your estimator is scored against.

## The process

For each MLP in the evaluation:

1. The evaluator generates random inputs from a standard normal distribution: each neuron receives an independent N(0, 1) value.
2. The inputs are propagated through the MLP (matrix multiply + ReLU at each layer).
3. This is repeated for `ground_truth_samples` independent draws; the official evaluation datasets use **1,000,000,000** (see [Configuration](#configuration)).
4. The per-neuron mean across all samples is the ground truth for each layer.

On the local, on-the-fly path the evaluator runs this through flopscope under an effectively unlimited FLOP budget (`1e15`). The official datasets are baked separately on GPU with a torch backend: same N(0, 1) inputs, the same matmul + ReLU, just a faster executor. Either way, ground-truth compute is never charged against your FLOP budget.

The gap between what the evaluator spends and what you are allowed to spend is deliberate, and it is enormous. At the evaluation shape (width 1024, depth 16) one Monte-Carlo forward pass costs ~33.6M FLOPs, so the entire per-MLP budget of `2**41 = 2,199,023,255,552` FLOPs buys roughly **65,000** forward passes. Ground truth is baked from **1,000,000,000** of them, about 15,000x more sampling compute than any estimator is allowed to spend. Out-sampling the reference is not on the table; beating it structurally is the whole point.

## Ground truth has its own error

Because ground truth is estimated by sampling, it has finite precision. With k samples, the standard error of the mean is approximately:

    standard_error ≈ sigma / sqrt(k)

The official leaderboard datasets bake their ground truth with **N = 1,000,000,000 samples per MLP**, the same process as the public release [`arc-whestbench-public-2026`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026). The floor this puts on your `final_layer_mse` (what a *perfect* estimator would still incur) is `avg_variance / N`, where `avg_variance` is the dataset's measured per-neuron final-layer activation variance. The ranked score inherits that floor scaled by your compute multiplier, so at the 0.1 multiplier floor it is a tenth of that again.

On the Phase 2 release (`@v2-phase2`) that variance measures **0.0748** across the Mini split, putting the ground-truth MSE floor at **7.5e-11**, orders of magnitude below any meaningful estimator gap. Covariance propagation, the most accurate bundled example, measures `4.05e-06` on the same split, about 54,000x above the floor. (Running `whest run` *without* `--dataset` instead generates ground truth on the fly at the lower-precision local default of **200,000** samples. That is 5,000x fewer, so the floor rises by the same factor, to `3.7e-07`. That local floor is about 9% of covariance propagation's `4.05e-06`: on a strong estimator the local floor informs the number you see rather than dominating it, and on a weaker one it is invisible. Raise it with `--n-samples` for a quieter comparison; the graded score always comes from `--dataset`.) Your MSE never reaches exactly zero, but against the official datasets the target is effectively exact.

## What this means for your estimator

- A "perfect" estimator that exactly matches the theoretical means would still show nonzero MSE due to ground truth sampling noise.
- Against the official datasets (N = 1e9 samples) the ground-truth noise floor is `avg_variance / N`, or `7.5e-11` on the Phase 2 release. In practice you hit your estimator's *own* approximation error long before ground-truth noise matters: covariance propagation lands at `4.05e-06`, still ~54,000x above the floor.
- Local on-the-fly runs (`whest run` without `--dataset`) re-sample ground truth, so different `--seed` values give slightly different MLPs and scores; the official baked datasets are fixed, so the leaderboard's ground truth never changes.

## Configuration

The number of ground truth samples is set in the contest configuration (`ContestSpec`), which defines all evaluation parameters: width, depth, FLOP budget, number of MLPs, and ground truth sample count. You can override some of these via CLI flags (e.g., `--n-mlps`, `--flop-budget`, `--n-samples`).

- `ground_truth_samples`: forward passes used to estimate ground truth. `whest run` without `--dataset` generates these on the fly and defaults to **200,000** per MLP (override with `--n-samples`; upstream measures roughly 6 s per MLP at this shape); the official baked datasets use **N = 1,000,000,000**.

Higher values produce more accurate ground truth but take longer to compute.

## ➡️ Next step

- [Scoring Model](./scoring-model.md)
- [Problem Setup](./problem-setup.md)

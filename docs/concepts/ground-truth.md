# How Ground Truth Is Generated

> [← Documentation](../README.md)

This page explains how the evaluator computes the reference values your estimator is scored against.

## The process

For each MLP in the evaluation:

1. The evaluator generates random inputs from a standard normal distribution: each neuron receives an independent N(0, 1) value.
2. The inputs are propagated through the MLP (matrix multiply + ReLU at each layer).
3. This is repeated for `ground_truth_samples` independent draws — the official evaluation datasets use **1,000,000,000** (see [Configuration](#configuration)).
4. The per-neuron mean across all samples is the ground truth for each layer.

The evaluator uses flopscope for this computation, but under a very large FLOP budget (effectively unlimited). Ground truth computation is not constrained by the participant's FLOP budget.

## Ground truth has its own error

Because ground truth is estimated by sampling, it has finite precision. With k samples, the standard error of the mean is approximately:

    standard_error ≈ sigma / sqrt(k)

The official leaderboard datasets bake their ground truth with **N = 1,000,000,000 samples per MLP** — the same process as the public release, [`arc-whestbench-public-2026`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026). The floor this puts on your score (what a *perfect* estimator would still incur) is `avg_variance / N`, where `avg_variance` is the dataset's measured per-neuron final-layer activation variance (≈ 0.18). That works out to a ground-truth MSE floor of **≈ 2e-10** — orders of magnitude below any meaningful estimator gap (covariance propagation is ~`8.4e-5`, roughly 10^5× higher). (Running `whest run` *without* `--dataset` instead generates ground truth on the fly at the lower-precision local default of 2,560,000 samples, where the floor rises to ~`7e-8` — fine for quick iteration, but noisier.) Your MSE never reaches exactly zero, but against the official datasets the target is effectively exact.

## What this means for your estimator

- A "perfect" estimator that exactly matches the theoretical means would still show nonzero MSE due to ground truth sampling noise.
- Against the official datasets (N = 1e9 samples) the ground-truth noise floor is ~`2e-10` (= `avg_variance / N`), so in practice you hit your estimator's *own* approximation error long before ground-truth noise matters — a strong estimator like covariance propagation lands around `1e-5`–`1e-4`, still far above that floor.
- Local on-the-fly runs (`whest run` without `--dataset`) re-sample ground truth, so different `--seed` values give slightly different MLPs and scores; the official baked datasets are fixed, so the leaderboard's ground truth never changes.

## Configuration

The number of ground truth samples is set in the contest configuration (`ContestSpec`), which defines all evaluation parameters: width, depth, FLOP budget, number of MLPs, and ground truth sample count. You can override some of these via CLI flags (e.g., `--n-mlps`, `--flop-budget`, `--n-samples`).

- `ground_truth_samples`: forward passes used to estimate ground truth. `whest run` without `--dataset` generates these on the fly and defaults to `100 * 100 * 256 = 2,560,000`; the official baked datasets use **N = 1,000,000,000**.

Higher values produce more accurate ground truth but take longer to compute. The contest organizer balances this tradeoff.

## ➡️ Next step

- [Scoring Model](./scoring-model.md)
- [Problem Setup](./problem-setup.md)

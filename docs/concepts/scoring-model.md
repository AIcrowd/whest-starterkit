# Scoring Model

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page to understand how the leaderboard score is computed from your estimator's predictions.

## Pipeline at a glance

```
   ┌─────────────────────┐
   │  random MLP_m       │   one of M MLPs (default M=10)
   │  flop_budget (B)    │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────────────────┐
   │  your predict(mlp_m, budget)    │   runs inside flopscope.BudgetContext
   │  (flopscope counts every op)    │
   └──────────┬──────────────────────┘
              │
              ▼
     effective compute  C_m = F_m + λ·R_m  >  B ?
         /                                      \
     yes / budget exceeded                       \ no
        ▼                                          ▼
   ┌──────────────────────┐      ┌──────────────────────┐
   │ pred_m  := zeros     │      │ pred_m  := your array │
   │ mult_m  := 1.0       │      │ mult_m  :=            │
   │ (no compute discount)│      │   max(0.1, C_m / B)   │
   └──────────┬───────────┘      └──────────┬───────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
        ┌────────────────────────────────────────────────┐
        │ final_layer_mse_m = mean((pred_m − truth_m)²)   │
        │                     over the final layer        │
        │ adjusted_m        = final_layer_mse_m × mult_m  │
        └───────────────────────┬────────────────────────┘
                                │
                     (repeat for every MLP)
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │ adjusted_final_layer_score = mean_m(adjusted_m) │
        │     ← THIS is the leaderboard ranking metric    │
        │ final_layer_mse, all_layers_mse = mean raw MSE  │
        │     (diagnostics only — no multiplier)          │
        └────────────────────────────────────────────────┘

                       lower is better
```

## 📌 TL;DR

- Lower score is better.
- The leaderboard ranks on **`adjusted_final_layer_score`** — the final-layer MSE scaled by a compute multiplier `max(0.1, C_m / flop_budget)`, then averaged across MLPs.
- `final_layer_mse` and `all_layers_mse` are reported too, but as **raw diagnostics** — they are *not* the metric you are ranked on.
- The multiplier rewards using less compute, down to a 10× discount floor (reached at 10% budget use). Below that floor, only accuracy moves your score.
- If your estimator exceeds the FLOP budget, all predictions for that MLP are zeroed **and** the multiplier is forced to 1.0 — a failure is strictly worse than the cheapest valid submission.

## The core idea

The scoring model answers a specific question: **how accurately can your estimator predict expected neuron values, and how little compute does it spend doing so?**

Each estimator call is given a `flop_budget` — a cap on the floating-point operations it may perform, tracked analytically by flopscope. If the estimator stays within budget, its final-layer predictions are scored by MSE against Monte Carlo ground truth, and that MSE is then **scaled by how much of the budget you used** to form the leaderboard score. If it exceeds the budget, all predictions for that MLP are replaced with zeros and no compute discount is applied.

## How scoring works

For the configured FLOP budget `B`:

1. **Your estimator runs.** Your `predict(mlp, budget)` is called. flopscope counts every floating-point operation analytically (`F_m`); the harness also measures the residual wall-time bucket (`R_m`) — Python-side work that runs outside a flopscope kernel.
2. **Effective compute is formed.** `C_m = F_m + λ·R_m`, where λ (the residual-penalty rate, default `1e11` FLOPs/sec — the grader uses the configured contest rate) converts residual wall-time into FLOP-equivalents. Locally you can experiment with a different rate via `whest run --lambda-flops-per-second`; the leaderboard always uses the contest-configured value.
3. **Budget is checked.** If `C_m > B` (or flopscope trips mid-run, or a wall-time limit fires), all predictions for this MLP are replaced with zero vectors and the compute multiplier is forced to `1.0`.
4. **Raw accuracy is measured.** The final-layer mean squared error (MSE) between your predictions and Monte Carlo ground truth is computed — this is `final_layer_mse`, a diagnostic.
5. **Score is the budget-adjusted MSE.** The per-MLP score is `final_layer_mse × max(0.1, C_m / B)` — accuracy scaled by the share of budget you used, with the discount capped at 10×.

The leaderboard metric, **`adjusted_final_layer_score`**, is this per-MLP score averaged across all MLPs (multiplier forced to `1.0` wherever the budget was exceeded). Lower is better.

## The formula

The leaderboard ranks on a single metric, `adjusted_final_layer_score`: the
final-layer MSE scaled by a per-MLP compute multiplier, then averaged across
MLPs. **Lower is better.**

```
                              1   M
adjusted_final_layer_score = ─── ∑  final_layer_mse_m × max(0.1, C_m / B)
                              M  m=1

                     1   n
final_layer_mse_m  = ─── ∑  ( pred_m[d-1, i] − truth_m[d-1, i] )²
                     n  i=1
                        └──────── final-layer cells only ────────┘

  C_m = F_m + λ·R_m   effective compute: analytical FLOPs F_m plus residual
                      wall-time R_m converted at λ (default 1e11 FLOPs/sec)
  B   = flop_budget
  max(0.1, C_m / B)   compute multiplier — caps the discount at 10× (the 0.1
                      floor); forced to 1.0 for any MLP whose budget was exceeded
```

> **Why "score" and not "MSE"?** Once `final_layer_mse` is multiplied by the
> budget factor, the result is no longer a mean-squared-error — it is a derived
> ranking score. That is why the leaderboard field is named
> `adjusted_final_layer_score` (the `_score` suffix), while the raw diagnostics
> keep the `_mse` suffix.

The two raw MSEs are reported for diagnosis only — they carry **no** multiplier:

```
                   1   M    1   n
final_layer_mse = ─── ∑   ─── ∑  ( pred_m[d-1, i] − truth_m[d-1, i] )²
                   M  m=1   n  i=1
                            └──────── final-layer cells only ────────┘

                   1   M    1     d-1   n
all_layers_mse  = ─── ∑   ─── ∑     ∑   ( pred_m[k, i] − truth_m[k, i] )²
                   M  m=1  d·n k=0   i=1
                            └────── all (depth × width) cells ───────┘

  M       = number of MLPs in the suite (default 10; --n-mlps overrides)
  d       = mlp.depth, n = mlp.width
  pred_m  = (depth, width) array your predict() returned for MLP m
  truth_m = Monte-Carlo ground-truth means for MLP m
            (replaced with zeros if your call exceeded flop_budget)
```

`adjusted_final_layer_score` is what the leaderboard ranks on. `all_layers_mse`
helps you diagnose whether your error concentrates in the final layer or
accumulates earlier — see also `best_mlp_adjusted_final_layer_score` and
`worst_mlp_adjusted_final_layer_score` in the
[score report](../reference/score-report-fields.md).

## Budget behavior

Your estimator receives a `budget` argument (the FLOP budget `B`). It is a fixed
hard cap for the run, so fixed-strategy estimators that always use the same
approach are a good default — as long as they stay within budget. Because the
score scales with `C_m / B`, spending **less** compute lowers (improves) your
score, but only until you hit the 0.1 floor at 10% budget use; below that,
further savings don't help and accuracy is the only lever left.

## Budget enforcement rules

The budget is enforced analytically, and your compute usage feeds the score multiplier:

- **Exceeded budget.** If your effective compute `C_m` exceeds `flop_budget` — whether flopscope trips mid-run on analytical FLOPs, or the post-hoc `C_m > B` check fires on residual wall-time — **all** predictions for that MLP are replaced with zeros and the multiplier is forced to `1.0`. This is a hard cutoff, not per-depth. (A failed MLP therefore scores 10× worse than the cheapest valid submission, which earns the 0.1 floor.)
- **Under budget.** Predictions are used as-is, and the multiplier `max(0.1, C_m / B)` rewards using less of the budget — down to a 10× discount at 10% utilization.
- **Multiplier floor.** Below 10% budget use the multiplier clamps at `0.1`, so there is no further reward for getting cheaper — accuracy is what remains.

## What a good score looks like

A score near zero means your predictions are accurate **and** you used little compute. A score well above zero means either your predictions are inaccurate, or your estimator exceeded the FLOP cap and was zeroed (with no compute discount).

Scores below what sampling would achieve at that budget indicate your structural approach is genuinely better than brute-force Monte Carlo. That is the research milestone this challenge targets.

## Practical tuning intuition

- Start with a safe method that consistently emits valid rows and stays within budget.
- Use `flop_budget` for hard-cap-aware implementation choices (not budget-time routing).
- Tune your implementation for the fixed budget profile you care about; the multiplier rewards staying well under budget, but only down to the 10% floor.
- Compare `final_layer_mse` and `all_layers_mse` in your reports to see which depths hurt your accuracy, and watch `mean_score_multiplier` to see how much the budget factor is scaling that accuracy.
- Use [evaluation datasets](../how-to/use-evaluation-datasets.md) to fix networks and ground truth across runs — this makes score comparisons meaningful and skips repeated sampling.

## Worked example

Suppose ground truth for a 3-neuron final layer is `[0.42, 0.38, 0.51]` and your estimator predicts `[0.40, 0.35, 0.55]`.

    final_layer_mse = mean([(0.40 - 0.42)^2, (0.35 - 0.38)^2, (0.55 - 0.51)^2])
                    = mean([0.0004, 0.0009, 0.0016])
                    = (0.0004 + 0.0009 + 0.0016) / 3
                    = 0.000967

That `0.000967` is this MLP's raw `final_layer_mse`. To get the per-MLP score that the leaderboard actually uses, scale it by the compute multiplier. If the call spent 30% of its budget (`C_m / B = 0.30`, so the multiplier is `max(0.1, 0.30) = 0.30`):

    adjusted_m = 0.000967 × 0.30 = 0.000290

The leaderboard `adjusted_final_layer_score` is the **mean of these per-MLP `adjusted_m` values across all MLPs** in the evaluation — a mean of means, not a sum. (Had the same call used ≤10% of budget, the multiplier would clamp at the 0.1 floor: `0.000967 × 0.1 = 0.0000967`.)

## Example estimator benchmarks

The table below shows the **raw-MSE diagnostics** from the bundled example estimators run against the **public release dataset** — [`arc-whestbench-public-2026`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026), `mini` split (100 MLPs, 256×32, N=1e9 baked ground truth) — at the 2.72e11 FLOP budget. These are the unscaled `final_layer_mse` / `all_layers_mse` values; the leaderboard `adjusted_final_layer_score` multiplies `final_layer_mse` by `max(0.1, C_m / budget)` (≤ 1.0). Every bundled example spends well under 1% of the budget, so each one bottoms out at the **0.1 floor** — its ranked score is exactly `final_layer_mse ÷ 10`. Use these as calibration points for your own estimator.

| Estimator | `final_layer_mse` | `all_layers_mse` | Approach |
|-----------|-----------|---------------|----------|
| `random_estimator` | ~0.75 | ~0.62 | Returns random values — the interface walkthrough. The bundled [`estimator.py`](../../estimator.py) at the repo root is the true (all-zeros) baseline (~0.91); running `uv run whest init <dir>` in a fresh directory produces the same template. |
| `mean_propagation` | ~9.5e-04 | ~8.2e-04 | Diagonal variance, O(depth x width^2), ~11M FLOPs. ~1000x better than the zeros baseline. |
| `covariance_propagation` | ~8.4e-05 | ~5.6e-05 | Full covariance, O(depth x width^3), ~1.6B FLOPs. ~11x better again than mean propagation. |

**How to read these numbers:**

- The **zeros baseline** (`estimator.py`, ~0.91) and the **random estimator** (~0.75) give you the "doing nothing" scale — their MSE reflects the natural magnitude of the ground-truth activations.
- **Mean propagation** is ~1000x more accurate than zeros — a huge improvement from a simple analytical formula at ~11M FLOPs (well under 1% of budget).
- **Covariance propagation** is another ~11x better, but costs O(width^3) per layer (~1.6B FLOPs at width=256/depth=32, still <1% of budget). The cubic cost grows fast — by around width≈1400 it would consume the whole 2.72e11 budget.
- The **leaderboard score** (`adjusted_final_layer_score`) is not shown directly: it scales each estimator's `final_layer_mse` by `max(0.1, C_m / budget)`. Every bundled example spends <1% of the budget, so all of them bottom out at the **0.1 floor** — each one's ranked score is exactly its `final_layer_mse ÷ 10`.

To reproduce: `uv run whest run --estimator examples/<NN>_<name>.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1` (e.g. `examples/02_mean_propagation.py`)

These numbers are reproducible: the `mini` split fixes the 100 MLPs and bakes ground truth at N=1e9, so re-running yields the same values (the `random_estimator` row uses `--seed 42`).

## ➡️ Next step

- [Score Report Fields](../reference/score-report-fields.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)

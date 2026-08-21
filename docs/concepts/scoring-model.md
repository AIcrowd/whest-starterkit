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
     compute used  C_m = F_m  >  B ?   or   residual R_m > 400 ms ?
         /                                      \
     yes / any cap blown                        \ no
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
- **Compute is analytical FLOPs, and nothing else: `C_m = F_m`.** The per-MLP budget is `B = 2**41 = 2,199,023,255,552` FLOPs.
- Residual wall-time is **not priced** into your score. It is capped instead: **400 ms per MLP**, and blowing the cap zeroes that MLP's predictions.
- `final_layer_mse` and `all_layers_mse` are reported too, but as **raw diagnostics** — they are *not* the metric you are ranked on.
- The multiplier rewards using less compute, down to a 10× discount floor (reached at 10% budget use). Below that floor, only accuracy moves your score.
- If your estimator blows any cap — the FLOP budget, the 400 ms residual cap, or the 120 s wall-clock cap — all predictions for that MLP are zeroed **and** the multiplier is forced to 1.0. A failure is strictly worse than the cheapest valid submission.

## The core idea

The scoring model answers a specific question: **how accurately can your estimator predict expected neuron values, and how little compute does it spend doing so?**

Each estimator call is given a `flop_budget` — a cap on the floating-point operations it may perform, tracked analytically by flopscope. If the estimator stays within budget, its final-layer predictions are scored by MSE against Monte Carlo ground truth, and that MSE is then **scaled by how much of the budget you used** to form the leaderboard score. If it blows the FLOP budget, or overruns the residual or wall-clock caps, all predictions for that MLP are replaced with zeros and no compute discount is applied.

## How scoring works

For the per-MLP FLOP budget `B = 2**41 = 2,199,023,255,552`:

1. **Your estimator runs.** Your `predict(mlp, budget)` is called. flopscope counts every floating-point operation analytically (`F_m`); the harness also measures the residual wall-time bucket (`R_m`) — Python-side work that runs outside a flopscope kernel.
2. **Compute is read off flopscope.** `C_m = F_m`. Only the operations flopscope counted are charged to you; wall-clock time is not converted into FLOPs.
3. **Caps are checked.** If `C_m > B` (or flopscope trips mid-run), if `R_m` exceeds the 400 ms residual cap, or if the 120 s wall-clock cap fires, all predictions for this MLP are replaced with zero vectors and the compute multiplier is forced to `1.0`.
4. **Raw accuracy is measured.** The final-layer mean squared error (MSE) between your predictions and Monte Carlo ground truth is computed — this is `final_layer_mse`, a diagnostic.
5. **Score is the budget-adjusted MSE.** The per-MLP score is `final_layer_mse × max(0.1, C_m / B)` — accuracy scaled by the share of budget you used, with the discount capped at 10×.

The leaderboard metric, **`adjusted_final_layer_score`**, is this per-MLP score averaged across all MLPs (multiplier forced to `1.0` wherever a cap was blown). Lower is better.

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

  C_m = F_m           compute used: the analytical FLOPs flopscope counted.
                      Residual wall-time does not enter C_m — it is capped
                      separately at 400 ms per MLP.
  B   = flop_budget   2**41 = 2,199,023,255,552 FLOPs per MLP
  max(0.1, C_m / B)   compute multiplier — caps the discount at 10× (the 0.1
                      floor); forced to 1.0 for any MLP that blew a cap
```

> **Changed in Phase 2.** Phase 1 ranked on *effective compute*
> `C_m = F_m + λ·R_m`, with λ = `1e11` FLOPs/sec converting residual wall-time
> into FLOP-equivalents — slow Python was simply expensive, and you could choose
> to pay for it. Phase 2 removes λ entirely: `C_m = F_m`, and residual wall-time
> is governed by a hard **400 ms** cap per MLP instead of a price. The residual
> bucket is there for plumbing, not for computation — see
> [Budget enforcement rules](#budget-enforcement-rules).

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
            (replaced with zeros if your call blew any cap)
  truth_m = Monte-Carlo ground-truth means for MLP m
```

`adjusted_final_layer_score` is what the leaderboard ranks on. `all_layers_mse`
helps you diagnose whether your error concentrates in the final layer or
accumulates earlier — see also `best_mlp_adjusted_final_layer_score` and
`worst_mlp_adjusted_final_layer_score` in the
[score report](../reference/score-report-fields.md).

## Budget behavior

Your estimator receives a `budget` argument (the FLOP budget `B`, `2**41`). It is
a fixed hard cap for the run, so fixed-strategy estimators that always use the
same approach are a good default — as long as they stay within budget. Because
the score scales with `C_m / B`, spending **less** compute lowers (improves) your
score, but only until you hit the 0.1 floor at 10% budget use; below that,
further savings don't help and accuracy is the only lever left.

## Budget enforcement rules

The FLOP budget is enforced analytically and feeds the score multiplier; the two
time caps are pass/fail and do not feed the multiplier at all:

- **Exceeded FLOP budget.** If `C_m` exceeds `flop_budget`, flopscope trips mid-run, **all** predictions for that MLP are replaced with zeros, and the multiplier is forced to `1.0`. This is a hard cutoff, not per-depth. (A failed MLP therefore scores 10× worse than the cheapest valid submission, which earns the 0.1 floor.)
- **Exceeded residual wall-time.** Residual wall-time `R_m` — Python-side work outside a flopscope kernel — is capped at **400 ms per MLP**. Overrun it and that MLP's predictions are zeroed and the multiplier forced to `1.0`, exactly as for a blown FLOP budget. The cap is not a price: there is no exchange rate that lets you buy residual time with score.
- **Exceeded wall-clock.** A **120 s** wall-clock cap per MLP catches runs that hang rather than overspend. Same consequence: zeros, multiplier `1.0`.
- **Under budget.** Predictions are used as-is, and the multiplier `max(0.1, C_m / B)` rewards using less of the budget — down to a 10× discount at 10% utilization.
- **Multiplier floor.** Below 10% budget use the multiplier clamps at `0.1`, so there is no further reward for getting cheaper — accuracy is what remains.

> **The residual bucket is for plumbing, not for computation.** Its 400 ms cap
> exists to absorb array marshalling, control flow and bookkeeping — not to give
> you a second, unmetered compute lane. Doing meaningful numerical work outside
> flopscope's accounting is a rules violation, not a trade-off you are allowed to
> make, and it is grounds for disqualification however cheap it looks on the
> leaderboard. If a legitimate estimator genuinely cannot fit its plumbing into
> 400 ms, write to [arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com) rather than working around the
> cap; the same address takes reports of ops you believe flopscope mis-prices.

## What a good score looks like

A score near zero means your predictions are accurate **and** you used little compute. A score well above zero means either your predictions are inaccurate, or your estimator blew a cap and was zeroed (with no compute discount).

Scores below what sampling would achieve at that budget indicate your structural approach is genuinely better than brute-force Monte Carlo. That is the research milestone this challenge targets.

## Practical tuning intuition

- Start with a safe method that consistently emits valid rows and stays within budget.
- Use `flop_budget` for hard-cap-aware implementation choices (not budget-time routing).
- Keep Python-side work small enough to clear the 400 ms residual cap comfortably. It buys you nothing on the score, and overrunning it costs you the whole MLP.
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

Two things calibrate an estimator: what it costs, and how accurate it is. Cost is
fixed by the Phase 2 shape and budget, so the first table below is exact. Accuracy
depends on the evaluation dataset, so the second table is still the Phase 1
measurement — take the ranking from it, not the absolute values.

### What the examples cost (width=1024, depth=16)

Measured with flopscope against the per-MLP budget `B = 2**41 = 2,199,023,255,552`:

| Estimator | FLOPs (`F_m`) | % of `B` | Multiplier |
|-----------|-----------|---------------|----------|
| [`random_estimator`](../../examples/01_random.py) | 131,072 | 0.000% | 0.1 (floor) |
| [`mean_propagation`](../../examples/02_mean_propagation.py) | 86,639,616 | 0.004% | 0.1 (floor) |
| [`covariance_propagation`](../../examples/03_covariance_propagation.py) | 51,709,240,799 | 2.351% | 0.1 (floor) |

Every bundled example lands under the 10% threshold, so all three bottom out at the **0.1 floor** — each one's ranked score is exactly its `final_layer_mse ÷ 10`. None of them has any compute discount left to gain; accuracy is the only lever they have.

Covariance propagation costs **597x** what mean propagation costs at this shape (it was ~160x at the Phase 1 shape — quadrupling the width grows the O(width^3) term far faster than the O(width^2) one), and it still spends under 3% of `B`. Almost all of that is a single `fnp.einsum` per layer: 99.7% of its total. Because that term is cubic in width, the remaining headroom is smaller than it looks — at depth 16, a width of roughly **3,570** would consume the whole `2**41` budget.

Both figures above are what the examples cost *after* seeding their state at float32. flopscope bills float64 at twice the float32 rate, and `fnp.zeros`/`fnp.ones`/`fnp.eye` default to float64, so an estimator that leaves the default in place pays 2x on everything those arrays touch — even though `mlp.weights` is already float32. Measured on these two examples, the float32 seeding is worth **43.7%** and **50.0%** respectively, with the final-layer MSE unchanged to five significant figures. It is the single cheapest saving available, and it is easy to lose again: `flops.stats.norm.*` promotes float32 back to float64 to match scipy, so cast its result back. See [Performance Tips](../how-to/performance-tips.md).

### How accurate the examples are

Measured over the `mini` split of the public release — [`arc-whestbench-public-2026`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026) at `@v2-phase2` (100 MLPs, 1024×16, N=1e9 baked ground truth).

| Estimator | `final_layer_mse` | `all_layers_mse` | Approach |
|-----------|-----------|---------------|----------|
| `random_estimator` | 0.6856 | 0.5390 | Returns random values — the interface walkthrough. The bundled [`estimator.py`](../../estimator.py) at the repo root is the true (all-zeros) baseline (0.9095); running `uv run whest init <dir>` in a fresh directory produces the same template. |
| `mean_propagation` | 2.216e-04 | 1.611e-04 | Diagonal variance, O(depth x width^2). **4,104x** better than the zeros baseline. |
| `covariance_propagation` | 4.051e-06 | 2.229e-06 | Full covariance, O(depth x width^3). **54.7x** better again than mean propagation. |

**How to read these numbers:**

- The **zeros baseline** (`estimator.py`, 0.9095) and the **random estimator** (0.6856) give you the "doing nothing" scale — their MSE reflects the natural magnitude of the ground-truth activations.
- **Mean propagation** is 4,104x more accurate than zeros — a huge improvement from a simple analytical formula, and at the Phase 2 shape it costs 86,639,616 FLOPs, 0.004% of budget.
- **Covariance propagation** is another 54.7x better, but costs O(width^3) per layer — 51,709,240,799 FLOPs at 1024×16, 2.351% of budget.
- The **leaderboard score** (`adjusted_final_layer_score`) is not shown directly: it scales each estimator's `final_layer_mse` by `max(0.1, C_m / budget)`. Every bundled example spends at most 2.351% of the budget — still below the 10% floor threshold — so all of them bottom out at the **0.1 floor** — each one's ranked score is exactly its `final_layer_mse ÷ 10`.

To reproduce the accuracy table: `uv run whest run --estimator examples/<NN>_<name>.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2` (e.g. `examples/02_mean_propagation.py`)

These numbers are reproducible: the `mini` split fixes the 100 MLPs and bakes ground truth at N=1e9, so re-running yields the same values (the `random_estimator` row uses `--seed 42`).

## ➡️ Next step

- [Score Report Fields](../reference/score-report-fields.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)

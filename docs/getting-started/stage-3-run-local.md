# Stage 3: Run on the Public Set (In-Process Harness)

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · [2](stage-2-validate.md) · **3** · [4](stage-4-run-subprocess.md) · [5](stage-5-package.md)

Stage 2 confirms the contract. Stage 3 runs the **real scoring pipeline** (the same one the grader uses) against the **public Mini split** — 100 fixed MLPs with baked N=1e9 ground truth — and in-process, so you can drop `import pdb; pdb.set_trace()` anywhere in `predict()` and step through it.

## 🚀 Run it

```bash
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner local
```

`--split mini` selects the 100-MLP Mini split (it's the default split, so you can omit `--split`); `local` is the default runner, so you can omit `--runner local` too. Ground truth is precomputed at N=1e9, so there's no sampling step — after the first download (cached) later runs reuse it with no re-download — the `mini` split is a **7.03 GB** download, cached after the first call (8x Phase 1's 0.86 GB — one MLP's weights are now 16 x 1024 x 1024 float32; see [Use Evaluation Datasets](../how-to/use-evaluation-datasets.md)). The FLOP budget is `2**41` (2,199,023,255,552 FLOPs per MLP) and the MLP shape is the competition size (width=1024, depth=16 — Phase 1 used 256×32 at a `2.72e11` budget). *(Omit `--dataset` and `whest run` instead generates a fresh random 10-MLP suite on the fly, computing ground truth with 2,560,000 Monte-Carlo samples — slower and not reproducible. Fine for a quick `pdb` poke; use the Mini split for real scoring.)*

> **The dataset is currently the only way to get the Phase 2 shape here.** `whest run`
> takes `--flop-budget`, `--wall-time-limit` and `--residual-wall-time-limit`, so you can
> match three of the Phase 2 limits by hand — but it has **no `--width`/`--depth`**, and a
> generated suite is fixed at 256×32 on the pinned whestbench (0.15.0). If you do not
> have the evaluation dataset to hand, use **Stage 1** (`uv run python estimator.py`) for
> anything shape-sensitive: it builds its own 1024×16 MLP and needs no dataset. A generated
> `whest run` suite is still the right tool for exercising the *pipeline* — packaging,
> report fields, failure flags — just not for judging accuracy or cost at the Phase 2 shape.

You'll see a Rich-rendered report with five panels:

1. **Run Context** — estimator class, path, timestamps, `n_mlps`, `width`, `depth`, `flop_budget`.
2. **Hardware & Runtime** — host, OS, CPU, RAM, Python and NumPy versions. This makes the *analytical* part of a score reproducible across machines; `residual_wall_time_s` still depends on the machine, and it is capped at 400 ms per MLP rather than priced, which is why the panel records what that machine was (see [Is scoring hardware-dependent?](../troubleshooting/faq.md#is-scoring-hardware-dependent)).
3. **Sampling Budget Breakdown (Ground Truth)** — provenance/FLOPs for the reference ground truth (loaded from the baked dataset with `--dataset`; sampled locally otherwise).
4. **Estimator Budget Breakdown** — same fields for your `predict()` call(s).
5. **Final Score** — the headline metrics:

```
╭──────────────────────── Final Score ────────────────────────╮
│ Adjusted Final-Layer Score  [adjusted_final_layer_score]     │
│    ≈ 0.0910  ← primary score (what the leaderboard ranks on) │
│ Raw Final-Layer MSE         [final_layer_mse]      ≈ 0.9095  │
│ All-Layers MSE              [all_layers_mse]       ≈ 0.7636  │
│ ───────                                                      │
│ Best MLP   [best_mlp_adjusted_final_layer_score]   ≈ 0.0567  │
│ Worst MLP  [worst_mlp_adjusted_final_layer_score]  ≈ 0.1483  │
│ ───────                                                      │
│ Mean Score Multiplier     [mean_score_multiplier]  ≈ 0.10    │
│ Mean Compute Utilization  [mean_compute_utilization] ≈ 0     │
│ Failed MLPs               [n_failed_mlps]          0 of 100  │
╰ per-MLP score = final_layer_mse × max(0.1, C_m / flop_budget) ╯
```

With the zeros template, the **raw** MSE rows (`final_layer_mse` 0.9095, `all_layers_mse` 0.7636) reflect the natural variance of the ReLU activations. But the metric you are ranked on is `adjusted_final_layer_score`: the zeros template performs no metered work at all, so its multiplier sits at the 0.1 floor and the leaderboard score is `0.9095 × 0.1 ≈ 0.0910`. The per-MLP spread is wide even for a constant prediction — 0.0567 on the easiest MLP to 0.1483 on the hardest — because it tracks each MLP's own activation scale. Because the Mini split is fixed, these numbers are reproducible — no `--seed` needed. (`adjusted_final_layer_score` is the mean across MLPs of `final_layer_mse × max(0.1, C_m / flop_budget)`; the raw `final_layer_mse` / `all_layers_mse` carry no multiplier.) See [score-report-fields.md](../reference/score-report-fields.md) for the full schema.

## FLOP-budget callout: Stage 1 vs Stage 3

Stage 1's `local_engine.compare_against_monte_carlo` runs your `predict()` under `estimator_budget=2**41`, which is **the same** number Stage 3's `whest run` uses (`flop_budget=2**41`, 2,199,023,255,552 FLOPs per MLP). The two stages hold you to one cap, so a `predict()` that fits in Stage 1 fits at grading too, and budget exhaustion is unlikely to be why a Stage-1-good estimator scores differently in Stage 3.

## Why a different score than Stage 1?

Both stages use the same MLP shape (width=1024, depth=16). The numbers still differ because:

- **Stage 1** scores your estimator against **one fixed MLP** (`build_mlp(width=1024, depth=16, seed=0)`) and prints **raw MSE** as Monte-Carlo ground truth converges (10 → 100,000 samples).
- **Stage 3** scores the **100 MLPs of the public Mini split** against their baked N=1e9 ground truth, and reports the **budget-adjusted `adjusted_final_layer_score`** averaged across the suite — not raw MSE.

So Stage 3's headline number is averaged over 100 MLPs *and* scaled by the compute multiplier; expect it to differ from the single-MLP raw MSE you saw in Stage 1.

## Debugging

Because `--runner local` runs in-process, `pdb` works:

```python
def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
    import pdb; pdb.set_trace()
    ...
```

## ✅ Expected outcome

| Estimator | Raw `final_layer_mse` (Mini split, 100 MLPs) | Raw `all_layers_mse` |
|---|---|---|
| Zeros template | 0.9095 (the all-zeros accuracy floor) | 0.7636 |
| `01_random` | 0.6856 | 0.5390 |
| `02_mean_propagation` | 2.216e-04 | 1.611e-04 |
| `03_covariance_propagation` | 4.051e-06 | 2.229e-06 |

These are the **raw** final-layer MSEs (the accuracy signal). Your leaderboard `adjusted_final_layer_score` scales each by the compute multiplier `max(0.1, C_m / flop_budget)` — and since these all use at most ~2.4% of the budget (below the 10% floor threshold), the ranked number is exactly one-tenth of the value shown (the 0.1 floor).

Measured over all 100 Mini MLPs of `@v2-phase2` (1024x16, N=1e9 baked ground
truth). The gaps are much wider than at the Phase 1 shape: mean propagation is
**4,104x** more accurate than zeros here (it was ~1000x at 256x32), and
covariance propagation another **54.7x** beyond that (~11x at 256x32).

Stage 1 and Stage 3 will not line up, and that is expected rather than a bug:
Stage 1 scores one fixed MLP against on-the-fly Monte Carlo, Stage 3 scores the
100 fixed Mini MLPs against baked ground truth.

Full benchmark methodology in
[scoring-model.md](../concepts/scoring-model.md#example-estimator-benchmarks).

## ✅ When you're ready

Move on to [Stage 4: subprocess runner](stage-4-run-subprocess.md) for grader parity.

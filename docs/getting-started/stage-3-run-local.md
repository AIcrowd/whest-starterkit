# Stage 3: Run on the public set (in-process harness)

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · [2](stage-2-validate.md) · **3** · [4](stage-4-run-subprocess.md) · [5](stage-5-package.md)

Stage 2 confirms the contract. Stage 3 runs the **real scoring pipeline** (the same one the grader uses) against the **public Mini split**: 100 fixed MLPs with baked N=1e9 ground truth. It runs in-process, so you can add `import pdb; pdb.set_trace()` anywhere in `predict()` and step through it.

## 🚀 Run it

```bash
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner local
```

`--split mini` selects the 100-MLP Mini split (it's the default split, so you can omit `--split`); `local` is the default runner, so you can omit `--runner local` too. Ground truth is precomputed at N=1e9, so there's no sampling step. The `mini` split is a **7.03 GB** download, cached after the first call, so later runs reuse it with no re-download (8x Phase 1's 0.86 GB: one MLP's weights are now 16 x 1024 x 1024 float32; see [Use Evaluation Datasets](../how-to/use-evaluation-datasets.md)). The FLOP budget is `2**41` (2,199,023,255,552 FLOPs per MLP) and the MLP shape is the competition size (width=1024, depth=16). Phase 1 used 256×32 at a `2.72e11` budget; [Competition Rounds](../reference/rounds.md) compares every round.

> **No dataset to hand? You still get the Phase 2 shape.** Since whestbench 0.16.0 the
> graded round *is* the `whest run` default, so omitting `--dataset` still meters you
> exactly as the grader does: a generated 10-MLP suite at 1024x16 with
> `flop_budget=2**41`. Only the ground truth differs: 200,000 local Monte-Carlo draws
> per MLP (`--n-samples`) instead of the dataset's baked N=1e9. Both the MLPs and their
> sampled ground truth are redrawn on every run, so the score moves unless you pass
> `--seed`. Fine for a quick `pdb` session; use the Mini split for a number you can compare.
>
> | Graded limit | Value | Flag on `whest run` |
> |---|---|---|
> | MLP shape | 1024 x 16 | (the default; there is no `--width`/`--depth`) |
> | per-MLP FLOP budget | `2**41` (2,199,023,255,552) | `--flop-budget` |
> | per-`predict()` wall clock | 120 s | `--wall-time-limit` |
> | residual wall clock | 0.4 s, gated | `--residual-wall-time-limit` / `--no-residual-wall-time-limit` |
> | `setup()` timeout | 5 s | `--setup-timeout` |

You'll see a Rich-rendered report, a `WhestBench Report` banner followed by five panels:

1. **Run Context** — estimator class, path, timestamps, `n_mlps`, `width`, `depth`, `flop_budget`.
2. **Hardware & Runtime** — host, OS, CPU, RAM, Python and NumPy versions. This makes the *analytical* part of a score reproducible across machines; `residual_wall_time_s` still depends on the machine, and it is capped at 400 ms per MLP rather than priced, which is why the panel records what that machine was (see [Is scoring hardware-dependent?](../troubleshooting/faq.md#is-scoring-hardware-dependent)).
3. **Sampling Budget Breakdown (Ground Truth)** — provenance/FLOPs for the reference ground truth (loaded from the baked dataset with `--dataset`; sampled locally otherwise).
4. **Estimator Budget Breakdown** — same fields for your `predict()` call(s).
5. **Final Score** — the headline metrics:

```
╭──────────────────────────────── Final Score ─────────────────────────────────╮
│                                                                              │
│   metric                                           value                     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│   Adjusted Final-Layer Score                    9.10e-02   ← primary score   │
│   [adjusted_final_layer_score]                                               │
│   Raw Final-Layer MSE [final_layer_mse]         9.10e-01                     │
│   All-Layers MSE [all_layers_mse]               7.64e-01                     │
│   ────────                                      ────────                     │
│   Best MLP                                      5.67e-02                     │
│   [best_mlp_adjusted_final_layer_score]                                      │
│   Worst MLP                                     1.48e-01                     │
│   [worst_mlp_adjusted_final_layer_score]                                     │
│   ────────                                      ────────                     │
│   Mean Score Multiplier                       0.10000000                     │
│   [mean_score_multiplier]                                                    │
│   Mean Compute Utilization                    0.00000001                     │
│   [mean_compute_utilization]                                                 │
│   Failed MLPs [n_failed_mlps]                   0 of 100                     │
│                                                                              │
╰─ final_layer_mse: max(0.1, effective_compute/flop_budget) ───────────────────╯
```

With the zeros template, the **raw** MSE rows (`final_layer_mse` 0.9095, `all_layers_mse` 0.7636) reflect the natural variance of the ReLU activations. But the metric you are ranked on is `adjusted_final_layer_score`: the zeros template does almost no metered work (32,768 FLOPs per MLP, 1.5e-8 of the budget), so its multiplier is at the 0.1 floor and the leaderboard score is `0.9095 × 0.1 ≈ 0.0910`. (Stage 1 reports `estimator_flops = 0` for the same `predict()` because `local_engine` meters only the ops inside it; `whest run` also meters the array hand-off.) The per-MLP spread is wide even for a constant prediction (0.0567 on the easiest MLP to 0.1483 on the hardest) because it tracks each MLP's own activation scale. Because the Mini split is fixed, these numbers are reproducible with no `--seed` needed. (`adjusted_final_layer_score` is the mean across MLPs of `final_layer_mse × max(0.1, C_m / flop_budget)`; the raw `final_layer_mse` / `all_layers_mse` carry no multiplier.) See [score-report-fields.md](../reference/score-report-fields.md) for the full schema.

## FLOP-budget callout: Stage 1 vs Stage 3

Stage 1's `local_engine.compare_against_monte_carlo` runs your `predict()` under `estimator_budget=2**41`, which is **the same** number Stage 3's `whest run` uses (`flop_budget=2**41`, 2,199,023,255,552 FLOPs per MLP). The two stages hold you to one cap, so a `predict()` that fits in Stage 1 fits at grading too, and budget exhaustion is unlikely to be why a Stage-1-good estimator scores differently in Stage 3.

## Why a different score than Stage 1?

Both stages use the same MLP shape (width=1024, depth=16). The numbers still differ because:

- **Stage 1** scores your estimator against **one fixed MLP** (`build_mlp(width=1024, depth=16, seed=0)`) and prints **raw MSE** as Monte-Carlo ground truth converges (10 → 100,000 samples).
- **Stage 3** scores the **100 MLPs of the public Mini split** against their baked N=1e9 ground truth, and reports the **budget-adjusted `adjusted_final_layer_score`** averaged across the suite, not raw MSE.

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
| `03_covariance_propagation` | 4.050e-06 | 2.229e-06 |

These are the **raw** final-layer MSEs (the accuracy signal). Your leaderboard `adjusted_final_layer_score` scales each by the compute multiplier `max(0.1, C_m / flop_budget)`, and since these all use at most ~2.4% of the budget (below the 10% floor threshold), the ranked number is exactly one-tenth of the value shown (the 0.1 floor).

Measured over all 100 Mini MLPs of `@v2-phase2` (1024x16, N=1e9 baked ground
truth). The gaps are much wider than at the Phase 1 shape: mean propagation is
**4,104x** more accurate than zeros here (it was ~1000x at 256x32), and
covariance propagation another **54.7x** beyond that (~11x at 256x32). A number you
recorded before Phase 2 does not compare.
[Competition Rounds](../reference/rounds.md) explains why, and how to replay an
earlier round.

Stage 1 and Stage 3 do not match, and that is expected rather than a bug:
Stage 1 scores one fixed MLP against Monte-Carlo ground truth computed at run
time, and Stage 3 scores the 100 fixed Mini MLPs against baked ground truth.

Full benchmark methodology in
[scoring-model.md](../concepts/scoring-model.md#example-estimator-benchmarks).

## ✅ When you're ready

Move on to [Stage 4: subprocess runner](stage-4-run-subprocess.md) for grader parity.

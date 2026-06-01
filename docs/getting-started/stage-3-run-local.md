# Stage 3: Run Locally (In-Process Harness)

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · [2](stage-2-validate.md) · **3** · [4](stage-4-run-subprocess.md) · [5](stage-5-package.md)

Stage 2 confirms the contract. Stage 3 actually scores you against the same MLPs the grader uses — but in-process, so you can drop `import pdb; pdb.set_trace()` anywhere in `predict()` and step through it.

## 🚀 Run it

```bash
uv run whest run --estimator estimator.py --runner local
```

The default runner is `local` — you can omit `--runner local`. Defaults: `--n-mlps 10`, `--flop-budget 1e8` (100M), `--n-samples width*width*256`.

You'll see a Rich-rendered report with five panels:

1. **Run Context** — estimator class, path, timestamps, `n_mlps`, `width`, `depth`, `flop_budget`.
2. **Hardware & Runtime** — host, OS, CPU, RAM, Python and NumPy versions (so a leaderboard score is reproducible across machines).
3. **Sampling Budget Breakdown (Ground Truth)** — total FLOPs and time spent generating Monte-Carlo ground truth.
4. **Estimator Budget Breakdown** — same fields for your `predict()` call(s).
5. **Final Score** — the headline metrics:

```
╭──────────────── Final Score ────────────────╮
│  Primary Score    [primary_score]    ≈ 0.5  │
│  Secondary Score  [secondary_score]  ≈ 0.5  │
│  Best MLP Score   [best_mlp_score]   ≈ 0.5  │
│  Worst MLP Score  [worst_mlp_score]  ≈ 0.5  │
╰─ lower MSE is better; primary = mean MSE  ──╯
```

With the zeros template, all four scores hover around 0.5 — the natural variance of the ReLU activations. For deterministic numbers you can use `--seed 42`. (`primary_score` is the mean across MLPs of the final-layer MSE; `secondary_score` is the mean across MLPs of the all-layer MSE.) See [score-report-fields.md](../reference/score-report-fields.md) for the full schema.

## FLOP-budget callout: Stage 1 vs Stage 3

Stage 1's `local_engine.compare_against_monte_carlo` uses `estimator_budget=1e9` (more headroom for prototyping). Stage 3's default is `flop_budget=1e8` (the grader default). If your estimator's cost grows fast (e.g. covariance propagation at large widths), your Stage 3 score may be worse than Stage 1 — try `--flop-budget 1e9` to confirm before optimizing.

## Why a different MSE than Stage 1?

Stage 1 uses one fixed MLP (`build_mlp(width=32, depth=6, seed=0)`). Stage 3 generates a fresh suite of random MLPs at width=100, depth=16 (or loads a pre-created dataset via `--dataset`). Lower variance per MLP, but the average is what counts.

## Debugging

Because `--runner local` runs in-process, `pdb` works:

```python
def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
    import pdb; pdb.set_trace()
    ...
```

## ✅ Expected outcome

| Estimator | Typical `primary_score` (default settings) |
|---|---|
| Zeros template | ~0.5 (the all-zeros floor) |
| `02_mean_propagation` | ~0.004 |
| `03_covariance_propagation` | ~0.0003 |

(Same ballpark as the Stage 1 table because the math is the same; numbers
shift slightly because Stage 3 uses width=100 / depth=16 instead of
Stage 1's width=32 / depth=6.) Full benchmark methodology in
[scoring-model.md](../concepts/scoring-model.md#example-estimator-benchmarks).

## ✅ When you're ready

Move on to [Stage 4: subprocess runner](stage-4-run-subprocess.md) for grader parity.

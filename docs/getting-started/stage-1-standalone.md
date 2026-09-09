# Stage 1: Iterate locally (just `flopscope`)

> [← Tutorial](README.md)

> Ladder: **1** · [2](stage-2-validate.md) · [3](stage-3-run-local.md) · [4](stage-4-run-subprocess.md) · [5](stage-5-package.md)

"*just `flopscope`*" means: **no `whest` CLI required**. You run `python estimator.py` and the bundled [`local_engine.py`](../../local_engine.py) constructs an MLP, calls your `predict()` inside a `flopscope.BudgetContext`, and sweeps Monte-Carlo sample counts to print a FLOPs-vs-MSE table. The `whestbench.BaseEstimator` and `whestbench.MLP` types you'll see imported are the participant-facing types: a plain base class and a plain dataclass. Importing them *does* pull in the whestbench package and its `datasets`/`pyarrow` dependencies (about half a second of startup); what Stage 1 avoids is the CLI, not the import.

Iterate here until `predict()` converges, then go to Stage 2 to confirm the contract.

## 🚀 Run it

```bash
uv run python estimator.py
```

You should see a table like:

```
--- Your estimator ---
MLP: width=1024 depth=16 seed=0  (MC sampling seed=0)

Each row compares the same estimator prediction against an independent MC reference.
The MSE includes reference sampling noise; it is not the sampler's error against ground truth.

 n_samples |    sampling_flops | estimator_flops | all_layers_mse | final_layer_mse
-----------------------------------------------------------------------------------
        10 |       336,439,296 |               0 |       0.756302 |         1.10035
       100 |     3,363,803,136 |               0 |        0.73602 |         1.11441
     1,000 |    33,637,441,536 |               0 |       0.740319 |         1.12288
    10,000 |   336,373,825,536 |               0 |        0.74489 |         1.12936
   100,000 | 3,363,737,665,536 |               0 |       0.745042 |         1.12934
```

Two MSE columns. `final_layer_mse`, the right-hand one, is what the grader
ranks you on; `all_layers_mse` is whestbench's secondary metric, and it is
less strict.

The stub `predict()` returns all zeros, so `estimator_flops` is `0` and both
columns approach the average squared true activation means. The estimator runs
once: increasing `n_samples` improves the **reference**, not your prediction.
Each row uses fresh inputs, so adjacent MSE values can fluctuate. In expectation,
reference sampling noise contributes a term proportional to `1/n_samples`,
leaving your estimator's own error as the reference converges. That limiting
error is what you are trying to lower. To see the trend, run
`--baseline mean_propagation`: `all_layers_mse` runs 0.022622, 0.00221052,
0.000416999, 0.000189649, 0.000169955 — about a 10.2x drop over the first
decade, then flattening as estimator error dominates.

## Edit `predict()`

Open [estimator.py](../../estimator.py). The body of `predict()` returns all zeros; replace it with your idea. The template already imports `flopscope.numpy as fnp`, so any array op you write through `fnp` (or via Python operators on `fnp` arrays) is FLOP-counted automatically. If you also need the budget API itself (`flops.current_budget()`, `flops.BudgetContext`), add `import flopscope as flops` at the top; the template does not. See the [Flopscope Primer](../reference/flopscope-primer.md). Re-run; the MSE columns tell you how close you are, and `estimator_flops` shows what your math cost.

## Compare against a baseline

```bash
uv run python estimator.py --baseline mean_propagation
```

This loads `examples/02_mean_propagation.py` and runs both estimators on the same MLP.

## ✅ Expected outcome

On the default MLP, `n_samples=100,000` row:

| Estimator | `final_layer_mse` (ranked) | `all_layers_mse` | Status |
|---|---|---|---|
| Zeros template (default) | 1.12934 | 0.745042 | baseline; average squared activation means |
| `--baseline mean_propagation` | 0.000302775 | 0.000169955 | ~3,700x better on the ranked metric; first-order analytical |
| `--baseline covariance_propagation` | 5.84779e-06 | 4.5582e-06 | ~52x better than mean; tracks neuron correlations |

You're ready for Stage 2 once your estimator's MSE is below
the zeros floor and `estimator_flops` stays under the per-MLP budget.
`local_engine` applies the grader's own `2**41` (2,199,023,255,552 FLOPs), so
Stage 1 and Stage 3 hold you to exactly the same cap.

## ✅ When you're ready

Move on to [Stage 2: validate the contract](stage-2-validate.md).

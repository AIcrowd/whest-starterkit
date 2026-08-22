# Stage 1: Iterate Locally (Just `flopscope`)

> [← Tutorial](README.md)

> Ladder: **1** · [2](stage-2-validate.md) · [3](stage-3-run-local.md) · [4](stage-4-run-subprocess.md) · [5](stage-5-package.md)

"*Just `flopscope`*" means: **no `whest` CLI required**. You run `python estimator.py` and the bundled [`local_engine.py`](../../local_engine.py) constructs an MLP, calls your `predict()` inside a `flopscope.BudgetContext`, and sweeps Monte-Carlo sample counts to print a FLOPs-vs-MSE table. The `whestbench.BaseEstimator` and `whestbench.MLP` types you'll see imported are the participant-facing types: a plain base class and a plain dataclass. Importing them *does* pull in the whestbench package and its `datasets`/`pyarrow` dependencies (about half a second of startup); what Stage 1 avoids is the CLI, not the import.

Iterate here until `predict()` converges, then climb to Stage 2 to confirm the contract.

## 🚀 Run it

```bash
uv run python estimator.py
```

You should see a table like:

```
--- Your estimator ---
MLP: width=1024 depth=16 seed=0  (MC sampling seed=0)

 n_samples |    sampling_flops | estimator_flops | all_layers_mse | final_layer_mse
-----------------------------------------------------------------------------------
        10 |       336,439,296 |               0 |       0.744041 |        1.065473
       100 |     3,363,803,136 |               0 |       0.744617 |        1.123959
     1,000 |    33,637,441,536 |               0 |       0.745814 |        1.131387
    10,000 |   336,373,825,536 |               0 |       0.746173 |        1.131690
   100,000 | 3,363,737,665,536 |               0 |       0.745660 |        1.130490
```

Two MSE columns. `final_layer_mse`, the right-hand one, is what the grader
ranks you on; `all_layers_mse` is whestbench's more forgiving secondary
metric.

The stub `predict()` returns all zeros, so `estimator_flops` is `0` and both
columns plateau at the variance of the true outputs. Once you put real math in
`predict()`, they come alive and the MSE should shrink roughly as
`1/n_samples` (a 10x sample increase buys about 10x MSE, not about 3x,
because MSE is a squared error) until it flattens out at your estimator's own
error. That plateau is the number you are actually trying to lower; the rows
before it are just the Monte-Carlo reference getting quiet. You can watch it
happen with `--baseline mean_propagation`: `all_layers_mse` runs 0.020202,
0.001866, 0.000398, 0.000195, 0.000171 — a 10.8x drop over the first decade,
then flattening as bias takes over.

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
| Zeros template (default) | 1.130490 | 0.745660 | floor; natural variance of the activations |
| `--baseline mean_propagation` | 0.000300 | 0.000171 | ~3,800x better on the ranked metric; first-order analytical |
| `--baseline covariance_propagation` | 0.000004 | 0.000004 | ~72x better than mean; tracks neuron correlations |

You're ready for Stage 2 once your estimator's MSE is comfortably below
the zeros floor and `estimator_flops` stays under the per-MLP budget.
`local_engine` applies the grader's own `2**41` (2,199,023,255,552 FLOPs), so
Stage 1 and Stage 3 hold you to exactly the same cap.

## ✅ When you're ready

Move on to [Stage 2: validate the contract](stage-2-validate.md).

# Stage 1: Iterate Locally (Just `flopscope`)

> [← Tutorial](README.md)

> Ladder: **1** · [2](stage-2-validate.md) · [3](stage-3-run-local.md) · [4](stage-4-run-subprocess.md) · [5](stage-5-package.md)

"*Just `flopscope`*" means: **no `whest` CLI required**. You run `python estimator.py` and the bundled [`local_engine.py`](../../local_engine.py) constructs an MLP, calls your `predict()` inside a `flopscope.BudgetContext`, and sweeps Monte-Carlo sample counts to print a FLOPs-vs-MSE table. The `whestbench.BaseEstimator` and `whestbench.MLP` types you'll see imported are just the shared dataclasses — they don't pull in the harness.

Iterate here until `predict()` converges, then climb to Stage 2 to confirm the contract.

## 🚀 Run it

```bash
uv run python estimator.py
```

You should see a table like:

```
--- Your estimator ---
MLP: width=1024 depth=16 seed=0

 n_samples | sampling_flops | estimator_flops |        MSE
----------------------------------------------------------
        10 |    336,439,296 |               0 |   0.744041
       100 |  3,363,803,136 |               0 |   0.744617
     1,000 | 33,637,441,536 |               0 |   0.745814
    10,000 | 336,373,825,536 |               0 |   0.746173
   100,000 | 3,363,737,665,536 |               0 |   0.745660
```

The stub `predict()` returns all zeros, so `estimator_flops` is `0` and the MSE
plateaus at the variance of the true outputs — once you put real math in
`predict()`, both columns come alive and the MSE should shrink roughly as
`1/sqrt(n_samples)` (Monte Carlo converging to your estimator's answer).

## Edit `predict()`

Open [estimator.py](../../estimator.py). The body of `predict()` returns all zeros — replace it with your idea. The template already imports `flopscope as flops` and `flopscope.numpy as fnp`, so any array op you write through `fnp` (or via Python operators on `fnp` arrays) is FLOP-counted automatically. Re-run; the MSE column tells you how close you are, and `estimator_flops` shows what your math cost.

## Compare against a baseline

```bash
uv run python estimator.py --baseline mean_propagation
```

This loads `examples/02_mean_propagation.py` and runs both estimators on the same MLP.

## ✅ Expected outcome

| Estimator | MSE on the default MLP (`n_samples=100,000` row) | Status |
|---|---|---|
| Zeros template (default) | ~0.75 | floor — natural variance of the activations |
| `--baseline mean_propagation` | ~0.00017 | ~4,400x better; first-order analytical |
| `--baseline covariance_propagation` | ~0.000004 | ~43x better than mean; tracks neuron correlations |

You're ready for Stage 2 once your estimator's MSE is comfortably below
the zeros floor and `estimator_flops` stays under the per-MLP budget —
`local_engine` applies the grader's own `2**41` (2,199,023,255,552 FLOPs), so
Stage 1 and Stage 3 hold you to exactly the same cap.

## ✅ When you're ready

Move on to [Stage 2: validate the contract](stage-2-validate.md).

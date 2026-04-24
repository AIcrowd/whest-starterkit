# Stage 1: Iterate Locally (Just `whest`)

You don't need the `whest` CLI to start. The `local_engine.py` in this repo gives you the same MLP factory and Monte-Carlo helpers the harness uses internally — wired up so you can iterate on `predict()` and see convergence in seconds.

## Run it

```bash
python estimator.py
```

You should see a table like:

```
MLP: width=32 depth=6 seed=0

 n_samples | sampling_flops | estimator_flops |        MSE
-----------------------------------------------------------
        10 |       614,400 |         24,576 |   0.013214
       100 |     6,144,000 |         24,576 |   0.001327
     1,000 |    61,440,000 |         24,576 |   0.000133
    10,000 |   614,400,000 |         24,576 |   0.000013
   100,000 | 6,144,000,000 |         24,576 |   0.000001
```

The MSE should shrink roughly as `1/sqrt(n_samples)` — that's Monte Carlo converging to your estimator's answer.

## Edit `predict()`

Open [estimator.py](../../estimator.py). The body of `predict()` returns all zeros — replace it with your idea. Re-run; the MSE column tells you how close you are.

## Compare against a baseline

```bash
python estimator.py --baseline mean_propagation
```

This loads `examples/02_mean_propagation.py` and runs both estimators on the same MLP.

## When you're ready

Move on to [Stage 2: validate the contract](stage-2-validate.md).

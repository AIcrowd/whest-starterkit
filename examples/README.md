# Examples — A Curriculum

Read in order. Each file is a complete, runnable Stage 1 estimator.

| File | Difficulty | Expected MSE (default MLP, n=100k) | What it teaches |
|---|---|---|---|
| [01_random.py](01_random.py) | introductory | ~0.29 (random baseline) | The `BaseEstimator` interface and the contract: `predict(mlp, budget) -> we.ndarray of shape (depth, width)` |
| [02_mean_propagation.py](02_mean_propagation.py) | easy | ~0.003 | First-order analytical: propagate per-neuron mean and diagonal variance through ReLU layers |
| [03_covariance_propagation.py](03_covariance_propagation.py) | medium | ~0.0007 | Track full covariance, not just diagonal variance — costlier but more accurate |
| [04_combined.py](04_combined.py) | advanced | ~0.0007 | Use the FLOP budget: switch strategy based on budget remaining (at width=32/depth=6 it falls back to covariance propagation, hence the matching MSE) |

## Run any example

```bash
uv run python examples/02_mean_propagation.py
```

## Compare against your estimator

```bash
uv run python estimator.py --baseline mean_propagation
```

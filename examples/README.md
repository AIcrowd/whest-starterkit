# Examples — A Curriculum

Read in order. Each file is a complete, runnable Stage 1 estimator.

| File | Difficulty | Expected `all_layers_mse` (default MLP, n=100k) | `predict()` FLOPs (share of budget) | What it teaches |
|---|---|---|---|---|
| [01_random.py](01_random.py) | introductory | ~0.52 (random baseline) | 131,072 (0.000%) | The `BaseEstimator` interface and the contract: `predict(mlp, budget) -> fnp.ndarray of shape (depth, width)` |
| [02_mean_propagation.py](02_mean_propagation.py) | easy | ~0.00017 | 86,639,616 (0.004%) | First-order analytical: propagate per-neuron mean and diagonal variance through ReLU layers |
| [03_covariance_propagation.py](03_covariance_propagation.py) | medium | ~0.000004 | 51,709,240,799 (2.351%) | Track full covariance, not just diagonal variance — costlier but more accurate |
| [04_shipped_weights.py](04_shipped_weights.py) | easy | n/a (zeros baseline) | 32,768 (0.000%) | Ship a precomputed `weights.npz` next to your estimator and load it via `submission_dir` in `setup()` |

The default MLP is the competition shape (width 1024, depth 16, seed 0), and the
budget share is against the per-MLP budget of `2**41` FLOPs.

The column above is `all_layers_mse`, whestbench's **secondary** metric, the more
forgiving of the two. The **ranked** metric is `final_layer_mse`, which the local
table prints in its last column; the grader scores
`adjusted_final_layer_score = final_layer_mse * max(0.1, C_m / B_m)`. At n=100k the
same three examples measure ~0.84, ~0.00030 and ~0.0000042 on the ranked metric, so
do not tune against `all_layers_mse` alone.

## Run any example

```bash
uv run python examples/02_mean_propagation.py
```

## Compare against your estimator

```bash
uv run python estimator.py --baseline mean_propagation
```

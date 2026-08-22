# How-to — recipes for specific tasks

> [← Documentation](../README.md)

Task-oriented guides. Each one answers "I want to do X — give me the steps and the common mistakes." Use these alongside the [Tutorial](../getting-started/) for a guided sequence and the [Reference](../reference/) for exact APIs.

## Writing and iterating

| Doc | When to read |
|---|---|
| [write-an-estimator.md](write-an-estimator.md) | Implementing your custom estimator from scratch: minimal structure, contract checklist, common first failure. |
| [inspect-mlp-structure.md](inspect-mlp-structure.md) | Traversing the `MLP` object: fields, weights, shapes. |
| [validate-run-package.md](validate-run-package.md) | The standard local loop: `validate` → `run` → `package`. Includes a "Useful flags" table covering `--seed`, `--n-samples`, `--profile`, `--max-threads`, `--setup-timeout`, and other `whest run` flags. |
| [use-evaluation-datasets.md](use-evaluation-datasets.md) | Pre-creating an evaluation dataset for fast, reproducible iteration. |

## Optimizing

| Doc | When to read |
|---|---|
| [algorithm-ideas.md](algorithm-ideas.md) | Survey of estimation strategies: Monte Carlo, mean propagation, covariance, and why one fixed strategy outperforms runtime routing at this budget. Plus open directions (low-rank, layer-adaptive, spectral, importance sampling, higher moments). |
| [manage-flop-budget.md](manage-flop-budget.md) | Where your FLOPs go and how to fit a tighter budget. Includes a line-by-line walkthrough of `examples/02_mean_propagation.py`. |
| [performance-tips.md](performance-tips.md) | Concrete patterns: matmul placement, free ops, diagonal vs covariance, env-var knobs. |

## Debugging and shipping

| Doc | When to read |
|---|---|
| [debugging-checklist.md](debugging-checklist.md) | Tiered checklist for "estimator runs but something feels wrong": Tier 0 pure-Python loop, Tier 1 sanity, Tier 2 correctness, Tier 3 optimization. |
| [pre-submission-checklist.md](pre-submission-checklist.md) | Checks to run before you click "submit" on AIcrowd, including the Phase 2 allowed-code rules and the 10-submissions-per-UTC-day cap. |
| [ship-weights.md](ship-weights.md) | Pre-compute offline and load weights in `setup()` via `submission_dir`; the boundary between permitted data and prohibited code; the 5 s setup budget; multi-file submissions; 50 MiB / 50 file caps; `.whestignore`; package preview and `--dry-run`. |

## ➡️ Where to look next

- Not sure whether a technique is allowed at all? → [Allowed Code](../concepts/allowed-code.md).
- Need the exact contract / report fields? → [Reference](../reference/).
- Estimator throws an error you don't recognize? → [Troubleshooting](../troubleshooting/).
- Working through the stages in order? → [Tutorial](../getting-started/).

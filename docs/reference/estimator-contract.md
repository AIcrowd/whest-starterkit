# Estimator Contract

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page when you need exact estimator I/O requirements.

## Required interface

`predict(self, mlp: MLP, budget: int) -> fnp.ndarray`

Optional lifecycle hooks:

- `setup(self, context: SetupContext) -> None`
- `teardown(self) -> None`

### Lifecycle

```
  Estimator()           ──▶  __init__         (cheap; no I/O, no compute)
       │
       ▼
  setup(context)        ──▶  one call before any predict()
       │                     • runs OUTSIDE any BudgetContext (off-budget)
       │                     • bounded by setup_timeout_s (default ~5s)
       │                     • good for: lookup tables, config loads,
       │                                  shape-independent precompute
       ▼
  predict(mlp_1, b)     ──▶  one call per MLP
  predict(mlp_2, b)            • runs INSIDE a BudgetContext
  ...                          • bounded by --flop-budget and (optionally)
  predict(mlp_M, b)              --wall-time-limit / --residual-wall-time-limit
       │
       ▼
  teardown()            ──▶  one call after all predict() calls
                             • cleanup of resources opened in setup()
```

`setup()` and `teardown()` are entirely optional — `examples/02_*` and
`examples/03_*` skip both. Define them when you have shape-agnostic
precompute that's expensive enough to be worth doing once. See
[FAQ: Can I precompute things in setup()?](../troubleshooting/faq.md#can-i-precompute-things-in-setup)
for budget rules.

### `SetupContext` fields

| Field | Type | Description |
|---|---|---|
| `width` | `int` | Neuron count for generated MLPs |
| `depth` | `int` | Number of layers per MLP |
| `flop_budget` | `int` | FLOP cap for the estimator |
| `api_version` | `str` | Contract version string |
| `scratch_dir` | `str \| None` | Optional writable directory for caching across calls (subprocess runner; otherwise typically `None`) |
| `submission_dir` | `str \| None` | Folder your submission was extracted into — locally, your estimator's folder; populated by `whest validate` / `whest run` and on the grader. Load shipped files (e.g. `weights.npz`) from here. See [how-to/ship-weights.md](../how-to/ship-weights.md). |

## Input object quick reference

| Object | Field | Meaning |
|---|---|---|
| `MLP` | `width` | Number of neurons per layer |
| `MLP` | `depth` | Number of weight matrices (layers) |
| `MLP` | `weights` | Ordered weight matrices, each `(width, width)` |

For traversal examples, see [Inspect and Traverse MLP Structure](../how-to/inspect-mlp-structure.md).

## Output requirements per `predict` call

| Requirement | Rule |
|---|---|
| Shape | Return a 2D array with shape `(mlp.depth, mlp.width)` |
| Numeric validity | Every value is finite |

## FLOP tracking

Your estimator must use flopscope primitives (`import flopscope as flops` and `import flopscope.numpy as fnp`) for all numerical computation. flopscope tracks FLOP usage analytically. If the total FLOPs across your entire `predict` call exceed `flop_budget`, all predictions for that MLP are replaced with zero vectors and your MSE for that MLP is computed against zeros.

## Failure semantics

The harness never crashes on a bad estimator. Every failure mode is
surfaced as report data so that one bad MLP doesn't take down the run.

| Failure | Behavior | Report field(s) surfacing it | Stage that catches it first |
|---|---|---|---|
| Wrong return shape (not `(mlp.depth, mlp.width)`) | predictions for this MLP zeroed | `per_mlp[i].error.details.{expected_shape, got_shape}` | Stage 2 (`whest validate`) |
| Wrong dtype (not a `flopscope.numpy.ndarray`) | predictions for this MLP zeroed | `per_mlp[i].error` with hint | Stage 2 |
| Non-finite values (NaN, Inf) | predictions for this MLP zeroed | `per_mlp[i].error.details.cause_hints` | Stage 2 |
| `predict()` raised an exception | predictions for this MLP zeroed; harness continues to the next MLP; CLI exits `1` and prints an "Estimator Errors" panel | `per_mlp[i].{error, error_code, traceback}`; `error_code` is the Python exception class name | Stage 3 (`whest run`) |
| Exceeded `flop_budget` | flopscope raises `BudgetExhaustedError` *before* the over-budget op runs; predictions zeroed | `per_mlp[i].budget_exhausted: true` | Stage 3 |
| Exceeded `--wall-time-limit` (`wall_time_limit_s`) | flopscope raises `TimeExhaustedError`; predictions zeroed | `per_mlp[i].time_exhausted: true` | Stage 3 (with `--wall-time-limit`) |
| Exceeded `--residual-wall-time-limit` | scoring layer (not flopscope) zeroes the predictions after `predict()` returns | `per_mlp[i].residual_wall_time_exhausted: true` | Stage 3 (with `--residual-wall-time-limit`) |

When `predict()` raises, the runner captures the exception, records the
class name in `error_code`, and forwards a formatted `traceback` (subprocess
runs forward it across the worker boundary). Use `--debug` to see
tracebacks inline; `--fail-fast` to halt at the first failure.

Predictions for the failed MLP are scored against zeros, and the compute
multiplier is forced to 1.0 (no discount), so the failure *does* hurt your
`adjusted_final_layer_score`. If you want the run to stop at the first problem
rather than score-against-zeros, use `--fail-fast`.

For the structured `error.details` schema, see
[score-report-fields.md](score-report-fields.md#per-mlp-fields).

## ➡️ Next step

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)

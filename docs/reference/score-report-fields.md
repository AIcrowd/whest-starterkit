# Score Report Fields

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page to interpret `whest run` output fields.

## Top-level fields

Typical report sections include:

- `schema_version`
- `mode`
- `run_meta`
- `run_config`
- `run_config.seed` (always present; `null` when `--seed` was omitted; the integer value passed to `--seed` otherwise). When set, this is also the value forwarded to `SetupContext.seed` for the estimator's `setup()` call. See [Estimator Contract: Reproducibility](./estimator-contract.md#reproducibility-under-the-grader-seed).
- `run_config.dataset` (present when `--dataset` is used)
- `results`

## Host metadata

`run_meta.host` is always an object. If you set `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1`, WhestBench still records cheap host fields and any values available through `psutil`, but fallback-backed fields such as `cpu_count_physical` and `ram_total_bytes` may be `null`.

## Core result fields

Inside `results`:

| Field | Description |
|---|---|
| `adjusted_final_layer_score` | **Leaderboard metric.** Suite mean of per-MLP `adjusted_final_layer_score = final_layer_mse × max(0.1, C_m / B)`; failure → × 1.0. Lower is better. |
| `final_layer_mse` | Raw final-layer MSE averaged across MLPs (no budget multiplier). Diagnostic. |
| `all_layers_mse` | Raw all-layers MSE averaged across MLPs (no budget multiplier). Diagnostic — reveals where approximation error accumulates. |
| `best_mlp_adjusted_final_layer_score` | Minimum per-MLP `adjusted_final_layer_score` across the suite. |
| `worst_mlp_adjusted_final_layer_score` | Maximum per-MLP `adjusted_final_layer_score` across the suite. |
| `mean_score_multiplier` | Mean of per-MLP `max(0.1, C_m / B)` (1.0 on failure). Bounded `[0.1, 1.0]`. |
| `mean_compute_utilization` | Mean of per-MLP `C_m / B`, **unclamped** — can exceed 1.0 when an MLP busted the cap. |
| `mean_effective_compute` | Mean of per-MLP `effective_compute = F_m + λ·R_m`. |
| `n_failed_mlps` | Count of MLPs with any failure flag or `error_code` set. |
| `failure_breakdown` | Dict with independent counts per failure flag: `budget_exhausted`, `time_exhausted`, `residual_wall_time_exhausted`, `combined_budget_exhausted`, `error`. Sums can exceed `n_failed_mlps` because one MLP can carry multiple flags. |
| `breakdowns` | Aggregate FLOP/time breakdowns keyed by section name. Includes `sampling` and `estimator`. |
| `per_mlp` | Array of per-MLP detail records (see below) |

### Per-MLP fields

Each entry in `per_mlp`:

| Field | Type | Description |
|---|---|---|
| `mlp_index` | `int` | Index of the MLP in the evaluation set |
| `flops_used` | `int` | Total FLOPs used by your estimator for this MLP (`F_m`) |
| `effective_compute` | `float` | `C_m = F_m + λ·R_m`. Combined FLOP-equivalent compute used by the estimator. |
| `adjusted_final_layer_score` | `float` | `s_m` — the per-MLP budget-adjusted score that flows into the suite mean. |
| `combined_budget_exhausted` | `bool` | Whether the post-hoc check `C_m > B` fired (predictions zeroed if true). |
| `budget_exhausted` | `bool` | Whether the estimator exceeded the FLOP budget (predictions zeroed if true) |
| `time_exhausted` | `bool` | Whether the estimator exceeded the wall-clock limit for this MLP (predictions zeroed if true) |
| `residual_wall_time_exhausted` | `bool` | Whether WhestBench judged residual wall time to exceed `residual_wall_time_limit_s` (predictions zeroed if true) |
| `wall_time_s` | `float` | Total elapsed wall-clock time measured for this MLP's estimator context |
| `flopscope_backend_time_s` | `float` | Wall time inside counted flopscope numpy kernels — the participant's actual numpy compute |
| `flopscope_overhead_time_s` | `float` | Wall time inside flopscope's own dispatch code (wrapper preambles, FLOP bookkeeping, namespace push/pop). Framework cost, not participant cost. |
| `residual_wall_time_s` | `float` | Wall time inside the predict context that is neither flopscope backend execution nor flopscope dispatch — i.e. participant Python (loops, control flow), GC, and Python-callback op time. As of flopscope 0.7.0, data-movement NumPy ops (concatenate, stack, tile, repeat, take, pad, …) count as `flopscope_backend_time_s`, not residual. `R_m` in the scoring formula. |
| `final_layer_mse` | `float` | MSE of your final-layer predictions vs ground truth (no multiplier) |
| `all_layers_mse` | `float` | MSE of your all-layer predictions vs ground truth (no multiplier) |
| `breakdowns` | `dict \| null` | Per-MLP breakdown container. Currently includes estimator-only data under `estimator`. Sampling is aggregate-only. |

If the estimator raised an error, the entry also includes:

| Field | Type | Description |
|---|---|---|
| `error` | `str` \| `dict` | Legacy string message, or structured object: `{"message": str, "details": object}` |
| `error_code` | `str` | Stable identifier: `PREDICT_ERROR` for a `RunnerError`, or the Python exception class name otherwise |
| `traceback` | `str \| null` | Formatted traceback string for the failure. Forwarded from the subprocess worker when `--runner subprocess`/`server` is used; captured locally otherwise. |

For structured `error` objects, `error.details` includes:

- `expected_shape`: `List[int]` with expected `(depth, width)`.
- `got_shape`: `List[int]` observed from estimator output.
- `cause_hints`: `List[str]` with user-facing hints.
- `hint`: short summary hint.

## Budget-adjusted scoring

The leaderboard ranks submissions by `adjusted_final_layer_score`, the suite mean of the budget-adjusted per-MLP score:

```
adjusted_final_layer_score = final_layer_mse × max(0.1, C_m / B)   for valid runs
adjusted_final_layer_score = final_layer_mse × 1.0                 for failures (no compute discount)

C_m = F_m + λ · R_m                       (effective compute, FLOPs + FLOP-equivalents)
λ   = residual-wall-time conversion rate (default 1e11 FLOPs/sec, contest-configured)
```

Where `F_m` is the analytical FLOPs counted by flopscope (`flops_used`), `R_m` is the residual wall-time bucket (`residual_wall_time_s` — neither flopscope-backend nor flopscope-overhead), and `B` is `flop_budget`. The `max(0.1, …)` floor caps the discount at 10× so an arbitrarily cheap-but-wrong submission cannot dominate the ranking.

> **Why "score" not "MSE"?** Once `final_layer_mse` is multiplied by the budget factor `max(0.1, C_m/B)`, the result is no longer a mean-squared-error — it is a derived ranking score (denoted `s_m`). The `_score` suffix in `adjusted_final_layer_score` reflects this; the raw diagnostics `final_layer_mse` and `all_layers_mse` keep the `_mse` suffix because they remain genuine MSEs.

## Time decomposition

Every `predict()` call satisfies a strict three-bucket identity:

```
wall_time_s = flopscope_backend_time_s + flopscope_overhead_time_s + residual_wall_time_s
```

- `flopscope_backend_time_s` — numpy kernels actually crunching numbers via `flopscope.numpy.*`.
- `flopscope_overhead_time_s` — flopscope's own dispatch (wrapper preambles, FLOP bookkeeping, namespace push/pop).
- `residual_wall_time_s` — participant Python (loops, control flow), GC, and Python-callback op time; as of flopscope 0.7.0, data-movement NumPy ops (concatenate, stack, tile, repeat, take, pad, …) count as `flopscope_backend_time_s`, not residual.

The decomposition holds at every level: per-MLP, aggregated across MLPs, and per namespace inside `breakdowns`.

## Breakdown containers

When namespace-aware flopscope data is available, WhestBench adds breakdown containers in
these places:

- `results.breakdowns.estimator` - aggregated estimator breakdown across all evaluated MLPs
- `results.breakdowns.sampling` - aggregated sampling breakdown across all evaluated MLPs
- `results.per_mlp[].breakdowns.estimator` - one normalized estimator breakdown per MLP

Namespace normalization rules:

- sampling work is namespaced under `sampling.*`
- unlabeled estimator work becomes `estimator.estimator-client`
- explicit estimator namespace `phase` becomes `estimator.phase`
- nested estimator namespace `phase.subphase` becomes `estimator.phase.subphase`

Each breakdown summary also includes timing totals:

- `flopscope_backend_time_s` - accumulated time inside counted flopscope operations
- `flopscope_overhead_time_s` - accumulated time inside flopscope's own dispatch
- `residual_wall_time_s` - participant Python (loops, control flow), GC, and Python-callback op time; as of flopscope 0.7.0, data-movement NumPy ops (concatenate, stack, tile, repeat, take, pad, …) count as `flopscope_backend_time_s`, not residual.

For `results.breakdowns.*`, those values are aggregated across all evaluated
MLPs.

## Interpretation guide

- `final_layer_mse` is your most actionable accuracy diagnostic — it directly drives `adjusted_final_layer_score`.
- `mean_compute_utilization` and `mean_score_multiplier` together tell you whether you're hitting the **0.1 multiplier floor**. If `mean_compute_utilization` is well below 0.1, you have headroom to spend more compute "for free" (more compute will not hurt your score until utilization rises above 0.1).
- `n_failed_mlps` and `failure_breakdown` should be `0` and all-zeros for a healthy submission. Any failure (budget bust, time bust, exception, wrong shape, non-finite) means the affected MLP scored `final_layer_mse_m × 1.0` (no compute discount).
- `budget_exhausted` is the first thing to check if your score is unexpectedly high — exceeded budget means your predictions were zeroed.
- `time_exhausted` means the estimator crossed the wall-clock limit configured through `wall_time_limit_s` / `--wall-time-limit`.
- `residual_wall_time_exhausted` means residual wall time crossed WhestBench's `residual_wall_time_limit_s` / `--residual-wall-time-limit`.
- `combined_budget_exhausted` means the **post-hoc** check `C_m > B` fired — even if flopscope didn't trip, your residual wall time pushed effective compute past the cap.
- `flops_used` vs `flop_budget` shows analytical headroom. But the **real** budget is `effective_compute / flop_budget` — that's what determines the multiplier.
- High `flopscope_backend_time_s` relative to wall: numpy compute is the dominant cost. Healthy for a numpy-heavy estimator.
- High `flopscope_overhead_time_s` relative to wall: many small ops are paying the per-call dispatch tax. Consider batching with larger numpy primitives.
- High `residual_wall_time_s` relative to wall: participant Python is the bottleneck (tight loops, per-element attribute access, calls into uninstrumented libraries). This bucket is what `λ·R_m` charges against your effective compute.
- `adjusted_final_layer_score` is the budget-adjusted score (≤ raw `final_layer_mse` mean since the multiplier is ≤ 1.0). A value close to the raw mean means you used near-full budget; a value close to 1/10 of the raw mean means you used ≤10% of budget and got the maximum discount.

## Dataset traceability fields

When using `whest run --dataset`, the report includes `run_config.dataset`:

| Field | Description |
|---|---|
| `path` | Path, id, or repository reference used for the dataset input |
| `sha256` | SHA-256 hash of the file for integrity |
| `seed` | RNG seed used to generate the dataset |
| `n_mlps` | Number of MLPs in the dataset |
| `seed_protocol` | Object with `name` and `version`. WhestBench currently requires `version = "3.0"`. |

### Dataset format compatibility

Datasets produced by `whest dataset bake` are written as **directory bundles** in the schema-3.0 layout used by HF Hub (a per-split parquet under `data/`, a `metadata.json`, and a rendered `README.md`). The `metadata.json` carries `schema_version` (currently `"3.0"`) and `seed_protocol.version` (currently `"3.0"`, name `whestbench_explicit_per_mlp_seeds`). WhestBench refuses to load a dataset whose schema or seed-protocol version it doesn't recognize, and the error message points at the modern bake command.

The 3.0 seed protocol stores the per-MLP input seed in the parquet `mlp_seed` column. Each estimator receives the participant-facing seed via `mlp.seed`, derived deterministically from the input seed — see [estimator-contract.md](./estimator-contract.md#reproducibility-under-the-grader-seed) for how to consume it.

> Historical note: pre-`schema_version: 3.0` releases shipped datasets as `.npz` files produced by `whest create-dataset`. That command was renamed to `whest dataset bake` when the layout moved to the multi-file HF-friendly form, and `.npz` datasets are no longer supported.

## ➡️ Next step

- [Scoring Model](../concepts/scoring-model.md)
- [CLI Reference](./cli-reference.md)

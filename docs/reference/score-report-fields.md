# Score Report Fields

## When to use this page

Use this page to interpret `whest run` output fields.

## Top-level fields

Typical report sections include:

- `schema_version`
- `mode`
- `run_meta`
- `run_config`
- `run_config.seed` (present when `whest run --seed` is provided)
- `run_config.dataset` (present when `--dataset` is used)
- `results`

## Host metadata

`run_meta.host` is always an object. If you set `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1`, WhestBench still records cheap host fields and any values available through `psutil`, but fallback-backed fields such as `cpu_count_physical` and `ram_total_bytes` may be `null`.

## Core result fields

Inside `results`:

| Field | Description |
|---|---|
| `primary_score` | Leaderboard metric — final-layer MSE averaged across MLPs. Lower is better. |
| `secondary_score` | All-layer MSE averaged across MLPs. Lower is better. |
| `breakdowns` | Aggregate FLOP/time breakdowns keyed by section name. Includes `sampling` and `estimator`. |
| `per_mlp` | Array of per-MLP detail records (see below) |

### Per-MLP fields

Each entry in `per_mlp`:

| Field | Type | Description |
|---|---|---|
| `mlp_index` | `int` | Index of the MLP in the evaluation set |
| `flops_used` | `int` | Total FLOPs used by your estimator for this MLP |
| `budget_exhausted` | `bool` | Whether the estimator exceeded the FLOP budget (predictions zeroed if true) |
| `time_exhausted` | `bool` | Whether the estimator exceeded the wall-clock limit for this MLP (predictions zeroed if true) |
| `untracked_time_exhausted` | `bool` | Whether WhestBench judged non-whest time to exceed `untracked_time_limit_s` (predictions zeroed if true) |
| `wall_time_s` | `float` | Total elapsed wall-clock time measured for this MLP's estimator context |
| `tracked_time_s` | `float` | Time spent inside counted whest operations for this MLP |
| `untracked_time_s` | `float` | `wall_time_s - tracked_time_s`, i.e. time outside counted whest operations |
| `final_mse` | `float` | MSE of your final-layer predictions vs ground truth |
| `all_layer_mse` | `float` | MSE of your all-layer predictions vs ground truth |
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

## Breakdown containers

When namespace-aware whest data is available, WhestBench adds breakdown containers in
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

- `tracked_time_s` - accumulated time inside counted whest operations
- `untracked_time_s` - elapsed time outside counted whest operations

For `results.breakdowns.*`, those values are aggregated across all evaluated
MLPs.

## Interpretation guide

- `final_mse` is your most actionable diagnostic — it directly drives `primary_score`.
- `budget_exhausted` is the first thing to check if your score is unexpectedly high — exceeded budget means your predictions were zeroed.
- `time_exhausted` means the estimator crossed the wall-clock limit configured through `wall_time_limit_s` / `--wall-time-limit`.
- `untracked_time_exhausted` means the non-whest portion of execution crossed WhestBench's `untracked_time_limit_s` / `--untracked-time-limit`.
- `flops_used` vs `flop_budget` shows how much headroom you have. If you are consistently near the cap, consider lighter methods.
- `tracked_time_s` vs `untracked_time_s` tells you whether time is dominated by counted whest ops or by Python / other library work outside whest.
- `primary_score` is raw MSE — compare across runs to see whether estimator changes are helping.

## Dataset traceability fields

When using `whest run --dataset`, the report includes `run_config.dataset`:

| Field | Description |
|---|---|
| `path` | Absolute path to the dataset file |
| `sha256` | SHA-256 hash of the file for integrity |
| `seed` | RNG seed used to generate the dataset |
| `n_mlps` | Number of MLPs in the dataset |

## Next step

- [Scoring Model](../concepts/scoring-model.md)
- [CLI Reference](./cli-reference.md)

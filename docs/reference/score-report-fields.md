# Score report fields

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page to interpret `whest run` output fields.

## Top-level fields

Typical report sections include:

- `schema_version`
- `whestbench_version` — the whestbench that produced this report (for example, `"0.16.0"`).
- `mode`
- `run_meta`
- `run_config`
- `run_config.seed` (always present; `null` when `--seed` was omitted; the integer value passed to `--seed` otherwise). When set, this is also the value forwarded to `SetupContext.seed` for the estimator's `setup()` call. See [Estimator Contract: Reproducibility](./estimator-contract.md#reproducibility-under-the-grader-seed).
- `run_config.dataset` (present when `--dataset` is used)
- `run_config.{n_mlps, width, depth, flop_budget, setup_timeout_s, wall_time_limit_s, residual_wall_time_limit_s, lambda_flops_per_second}` — the shape and the limits this run actually applied. Since whestbench 0.16.0 the CLI defaults *are* the Phase 2 values, so an unflagged local run applies the round's limits; read these fields to confirm which round's rulebook produced an older report, or to check a run where you did pass flags. The one default that is **not** a graded value is `--n-samples`: dataset-less runs draw 200,000 ground-truth Monte-Carlo samples per MLP, against the 1e9 draws baked into the graded dataset, and `--n-samples` is not recorded in `run_config`. See [Estimator Contract: Phase 2 limits](./estimator-contract.md#phase-2-limits).
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
| `per_layer_mse` | Per-layer MSE averaged across MLPs. `list[float]` of length `depth` (16 in Phase 2). `per_layer_mse[-1]` is `final_layer_mse`, and the mean of the list is `all_layers_mse` (to float32 precision). Shows *where* your approximation degrades: a rising tail means error compounds with depth. |
| `best_mlp_adjusted_final_layer_score` | Minimum per-MLP `adjusted_final_layer_score` across the suite. |
| `worst_mlp_adjusted_final_layer_score` | Maximum per-MLP `adjusted_final_layer_score` across the suite. |
| `mean_score_multiplier` | Mean of per-MLP `max(0.1, C_m / B)` (1.0 on failure). Bounded `[0.1, 1.0]`. |
| `mean_compute_utilization` | Mean of per-MLP `C_m / B`, **unclamped** — can exceed 1.0 when an MLP busted the cap. |
| `mean_effective_compute` | Mean of per-MLP `effective_compute` (`C_m`). In Phase 2 `C_m = F_m`, so this is the mean of `flops_used`. |
| `n_failed_mlps` | Count of MLPs with any failure flag or `error_code` set. |
| `failure_breakdown` | Dict with independent counts per failure flag: `budget_exhausted`, `time_exhausted`, `residual_wall_time_exhausted`, `combined_budget_exhausted`, `error`. Sums can exceed `n_failed_mlps` because one MLP can carry multiple flags. `combined_budget_exhausted` is `0` under the current round's λ = 0, since `C_m = F_m` is already capped by flopscope; it can only fire on a report re-scored with a non-zero `--lambda-flops-per-second`. |
| `breakdowns` | Aggregate FLOP/time breakdowns keyed by section name. Includes `sampling` and `estimator`. |
| `per_mlp` | Array of per-MLP detail records (see below) |

### Per-MLP fields

Each entry in `per_mlp`:

| Field | Type | Description |
|---|---|---|
| `mlp_index` | `int` | Index of the MLP in the evaluation set |
| `mlp_name` | `str` | Deterministic human-readable slug for this MLP (for example, `"megan-chang"`). Use it as a stable label in your own logs. |
| `flops_used` | `int` | Total FLOPs used by your estimator for this MLP (`F_m`) |
| `effective_compute` | `float` | `C_m = F_m + λ·R_m`. With the round's default λ = 0 this is exactly equal to `flops_used`; check `run_config.lambda_flops_per_second` if it isn't. |
| `adjusted_final_layer_score` | `float` | `s_m` — the per-MLP budget-adjusted score that flows into the suite mean. |
| `combined_budget_exhausted` | `bool` | Whether the post-hoc check `C_m > B` fired (predictions zeroed if true). Phase 2 scores `C_m = F_m`, which flopscope already caps at `B`, so on the grader there is nothing left for this to catch. |
| `budget_exhausted` | `bool` | Whether the estimator exceeded the FLOP budget (predictions zeroed if true) |
| `time_exhausted` | `bool` | Whether the estimator exceeded the wall-clock limit for this MLP — 120 s on the grader (predictions zeroed if true) |
| `residual_wall_time_exhausted` | `bool` | Whether WhestBench judged residual wall time to exceed `residual_wall_time_limit_s` — 400 ms on the grader (predictions zeroed if true) |
| `wall_time_s` | `float` | Total elapsed wall-clock time measured for this MLP's estimator context |
| `flopscope_backend_time_s` | `float` | Wall time inside counted flopscope numpy kernels — the participant's actual numpy compute |
| `flopscope_overhead_time_s` | `float` | Wall time inside flopscope's own dispatch code (wrapper preambles, FLOP bookkeeping, namespace push/pop). Framework cost, not participant cost. |
| `residual_wall_time_s` | `float` | Wall time inside the predict context that is neither flopscope backend execution nor flopscope dispatch — that is, participant Python (loops, control flow), GC, and Python-callback op time. As of flopscope 0.7.0, data-movement NumPy ops (concatenate, stack, tile, repeat, take, pad, …) count as `flopscope_backend_time_s`, not residual. Gated against the 400 ms residual cap; it does not enter the score (Phase 1 charged it as `λ·R_m`). |
| `final_layer_mse` | `float` | MSE of your final-layer predictions vs ground truth (no multiplier) |
| `all_layers_mse` | `float` | MSE of your all-layer predictions vs ground truth (no multiplier) |
| `per_layer_mse` | `list[float]` | Per-layer MSE for this MLP, length `depth`. |
| `breakdowns` | `dict \| null` | Per-MLP breakdown container. Currently includes estimator-only data under `estimator`. Sampling is aggregate-only. |
| `traceback` | `str \| null` | **Always present**; `null` on a clean run. Non-null when this MLP produced no real predictions — an estimator exception, or budget/time exhaustion. Forwarded from the worker under `--runner subprocess`/`server`, captured in-process otherwise. Test `traceback is not None`, not key presence. |

If the estimator raised an error, the entry also includes:

| Field | Type | Description |
|---|---|---|
| `error` | `str` \| `dict` | Legacy string message, or structured object: `{"message": str, "details": object}` |
| `error_code` | `str` | The `RunnerError`'s own code (for example, `PREDICT_ERROR`) when the failure crossed a runner boundary, or the Python exception class name when it was raised in-process. **Not stable across runners**: the same wrong-shape return reports `"ValueError"` under `--runner local` (the default) and `"PREDICT_ERROR"` under `--runner subprocess`. Branch on `error.details` or on the boolean failure flags, not on this string. |

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

C_m = F_m                                 (effective compute = the FLOPs flopscope counted)
B   = 2**41 = 2,199,023,255,552           (per-MLP FLOP budget)
```

Where `F_m` is the analytical FLOPs counted by flopscope (`flops_used`) and `B` is `flop_budget`. The `max(0.1, …)` floor caps the discount at 10× so an arbitrarily cheap-but-wrong submission cannot dominate the ranking.

> **Phase 1 priced residual time; Phase 2 caps it.** Phase 1 folded the residual
> wall-time bucket into the score as `C_m = F_m + λ·R_m`, with `λ` a
> contest-configured conversion rate (1e11 FLOP-equivalents per second of
> residual). Phase 2 drops `λ`: residual wall time is capped at 400 ms per MLP
> instead of billed, so the scored `C_m` is `F_m`.

### Which cost model produced this report

Every report records the regime it was scored under.
`run_config.lambda_flops_per_second` is `0.0` in the current round, so
`C_m = F_m` and `effective_compute` equals `flops_used` exactly, locally and on
the grader. A non-zero value means the run re-scored an earlier priced round
(`--lambda-flops-per-second 1e11 --no-residual-wall-time-limit
--wall-time-limit 60 --flop-budget 272000000000`), in which case
`effective_compute = F_m + λ·R_m` and `combined_budget_exhausted` can fire on
residual time alone.

> **Why "score" not "MSE"?** Once `final_layer_mse` is multiplied by the budget factor `max(0.1, C_m/B)`, the result is no longer a mean-squared-error; it is a derived ranking score (denoted `s_m`). The `_score` suffix in `adjusted_final_layer_score` reflects this; the raw diagnostics `final_layer_mse` and `all_layers_mse` keep the `_mse` suffix because they remain genuine MSEs.

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
- `results.breakdowns.sampling.time_source` - `"bake"` for `--dataset` runs (whestbench ≥ 0.13): the sampling timings were measured on the dataset bake machine, not in your run. The run-report panel says the same and no longer shows a sampling Residual Wall Time row; the estimator panel's residual row is the only live-measured residual time, and the only one the 400 ms cap is applied to.
- `results.per_mlp[].breakdowns.estimator` - one normalized estimator breakdown per MLP

Each breakdown object has this shape:

```
{ flop_budget, flops_used, flops_remaining,
  wall_time_s, flopscope_backend_time_s, flopscope_overhead_time_s, residual_wall_time_s,
  by_namespace: {
    "estimator.estimator-client": {
      flops_used, calls, flopscope_backend_time_s, flopscope_overhead_time_s,
      operations: {
        "matmul": { flop_cost, calls, flopscope_backend_time_s, flopscope_overhead_time_s },
        ...
      }
    }, ...
  } }
```

`operations` is where your budget actually went. Rank it with:

```bash
uv run python -c "import json; ops = json.load(open('report.json'))['results']['breakdowns']['estimator']['by_namespace']['estimator.estimator-client']['operations']; [print(f\"{v['flop_cost']:>15,} {v['calls']:>6}  {k}\") for k, v in sorted(ops.items(), key=lambda kv: -kv[1]['flop_cost'])]"
```

Per-namespace entries carry only the two flopscope timings; `residual_wall_time_s`
exists at the breakdown top level only.

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
- `mean_compute_utilization` and `mean_score_multiplier` together tell you whether you're hitting the **0.1 multiplier floor**. If `mean_compute_utilization` is well below 0.1, spending more compute does not change your score until utilization rises above 0.1.
- `n_failed_mlps` and `failure_breakdown` should be `0` and all-zeros for a healthy submission. Any failure (budget bust, time bust, exception, wrong shape, non-finite) means the affected MLP scored `final_layer_mse_m × 1.0` (no compute discount).
- If your score is unexpectedly high, check `budget_exhausted` first: an exceeded budget means your predictions were zeroed.
- `time_exhausted` means the estimator crossed the wall-clock limit configured through `wall_time_limit_s` / `--wall-time-limit` — 120 s per MLP on the grader.
- `residual_wall_time_exhausted` means residual wall time crossed WhestBench's `residual_wall_time_limit_s` / `--residual-wall-time-limit` — 400 ms per MLP on the grader.
- `combined_budget_exhausted` fires when the post-hoc check `C_m > B` trips because residual wall time pushed effective compute past the cap. Phase 2 scores `C_m = F_m`, so there is nothing left for it to catch.
- In Phase 2 your multiplier depends only on `flops_used` against `flop_budget`: the FLOPs flopscope counted are exactly what sets it. `effective_compute` is the same number while λ = 0.
- High `flopscope_backend_time_s` relative to wall: numpy compute is the dominant cost. Healthy for a numpy-heavy estimator.
- High `flopscope_overhead_time_s` relative to wall: many small ops are accumulating per-call dispatch overhead. Consider batching with larger numpy primitives.
- High `residual_wall_time_s` relative to wall: participant Python is the bottleneck (tight loops, per-element attribute access, calls into uninstrumented libraries). This bucket costs you no FLOPs, but it is capped at 400 ms per MLP and crossing the cap zeroes that MLP's predictions. It is there for plumbing (moving results around between flopscope calls), and doing real numerical work in it is a rules violation; see [Estimator Contract](./estimator-contract.md#phase-2-limits).
- `adjusted_final_layer_score` is the budget-adjusted score (≤ raw `final_layer_mse` mean since the multiplier is ≤ 1.0). A value close to the raw mean means you used near-full budget; a value close to 1/10 of the raw mean means you used ≤10% of budget and got the maximum discount.

## Dataset traceability fields

When using `whest run --dataset`, the report includes `run_config.dataset`:

| Field | Description |
|---|---|
| `path` | Path, id, or repository reference used for the dataset input |
| `sha256` | SHA-256 hash of the file for integrity |
| `seed` | RNG seed used to generate the dataset |
| `n_mlps` | Number of MLPs in the dataset |
| `seed_protocol` | Object describing how per-MLP seeds were derived. WhestBench 0.16.0 loads `version` `"3.0"` (`whestbench_explicit_per_mlp_seeds`, seeds stored verbatim in the parquet `mlp_seed` column) and `"4.0"` (`whestbench_kdf_per_mlp_seeds`, keyed BLAKE2b derivation); legacy `"2.0"` (`whestbench_seedsequence_hierarchy`) is still accepted. A 4.0 block additionally carries `kdf`, `salt_source` (`"metadata"` or `"env"`), `salt_digest`, and optionally `salt_id` / `salt`; when `salt_source` is `"env"` the salt arrives through the `WHEST_SEED_SALT` environment variable. |

### Dataset format compatibility

Datasets produced by `whest dataset bake` are written as **directory bundles** in the schema-3.0 layout used by HF Hub (a per-split parquet under `data/`, a `metadata.json`, and a rendered `README.md`). The `metadata.json` carries `schema_version` (currently `"3.0"`) and `seed_protocol.version` (currently `"3.0"`, name `whestbench_explicit_per_mlp_seeds`). WhestBench refuses to load a dataset whose schema or seed-protocol version it doesn't recognize, and the error message points at the modern bake command.

The 3.0 seed protocol stores the per-MLP input seed in the parquet `mlp_seed` column. Each estimator receives the participant-facing seed via `mlp.seed`, derived deterministically from the input seed — see [estimator-contract.md](./estimator-contract.md#reproducibility-under-the-grader-seed) for how to consume it.

> Historical note: pre-`schema_version: 3.0` releases shipped datasets as `.npz` files produced by `whest create-dataset`. That command was renamed to `whest dataset bake` when the layout moved to the multi-file HF-friendly form, and `.npz` datasets are no longer supported.

## ➡️ Next step

- [Scoring Model](../concepts/scoring-model.md)
- [CLI Reference](./cli-reference.md)

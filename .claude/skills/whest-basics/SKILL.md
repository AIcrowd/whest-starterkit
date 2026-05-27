---
name: whest-basics
description: How to use the `whest` CLI for scoring estimators in the AIcrowd Whitebox Estimation Challenge. Load when running `whest run`, parsing its output, building datasets for fast iteration, validating an estimator, packaging a submission, or writing automation around any of these. Covers the JSON schema, the dataset caching trick that makes loops fast, and the score formula.
---

# whest CLI — basic usage

The `whest` CLI is provided by the pinned `whestbench` dep in [pyproject.toml](../../../pyproject.toml). Run it with `uv run whest ...` from the repo root. All commands take a `--help` flag.

## The two commands that matter for automation

### 1. `whest run` — score an estimator

The canonical automation-friendly invocation:

```bash
uv run whest run \
  --estimator path/to/estimator.py \
  --runner subprocess \
  --format json \
  --dataset path/to/dataset_dir/
```

- `--runner subprocess` runs the estimator in a fresh Python subprocess. Use this for automation (no import bleed between runs, matches the grader closer than `--runner local`).
- `--format json` (alias `--json`) emits a machine-readable JSON blob on stdout. Without it, you get a Rich-formatted terminal report that is not safe to parse.
- `--dataset` reuses pre-computed ground truth. Without it, Monte Carlo ground truth is recomputed each run (~20s per MLP). With it, scoring 10 MLPs takes ~3 seconds. **Always use a dataset in loops.** The value is a baked dataset directory (or `hf://owner/repo[@rev]` for HF Hub).

Other useful flags:
- `--n-mlps N`: without `--dataset`, scores against N freshly-sampled MLPs (default 10). With `--dataset`, defaults to the full dataset size and is clamped if you ask for more than the dataset contains.
- `--seed S`: without `--dataset`, seeds both MLP generation and estimator setup. With `--dataset`, MLP seeds come from the dataset and this flag seeds only estimator `setup()`. Default omitted (estimator `ctx.seed` defaults to 0; `run_config.seed` is `null` in JSON).
- `--split SPLIT`: required for multi-split datasets, optional for single-split.

### 2. `whest dataset bake`: pre-compute ground truth

```bash
# Recommended local eval set. Low score noise, bakes in ~30s on a laptop.
uv run whest dataset bake --n-mlps 30 --n-samples 100000 --width 256 --depth 8 --output train_dataset/

# Quick smoke-test set. Noisier scores but bakes in ~2s; use for one-shot sanity checks.
uv run whest dataset bake --n-mlps 5 --n-samples 5000 --width 256 --depth 8 --output smoke_dataset/

# Reproducible held-out test set: pin the per-MLP seeds.
uv run whest dataset bake --n-mlps 30 --n-samples 100000 --width 256 --depth 8 --mlp-seeds test_seeds.json --output test_dataset/
```

- **For real comparisons, use 30 MLPs and `--n-samples 100000`.** Fewer MLPs or fewer samples adds noise that can swamp the score differences between good ideas.
- **The leaderboard grader uses 50 MLPs with `--n-samples 1_000_000_000` (10^9).** Local scores will track but not exactly match the leaderboard at lower sample counts.
- One-time cost, then every subsequent `whest run --dataset` is fast.
- `--output` is a **directory** that must not already exist. `--width` and `--depth` are required (no defaults).
- `--mlp-seeds path/to/seeds.json` lets you pin which MLPs go into the dataset by passing an explicit JSON array of N non-negative 63-bit ints. Omit it to auto-generate via `secrets.randbits(63)`. Use a stable seed list for held-out test sets so results are reproducible across machines.
- The baked directory carries a content hash that `whest run` echoes back in `run_config.dataset.sha256`, useful for tagging experiment logs.
- The historical `whest create-dataset` command no longer exists. If you see it in an old script, it's the same operation under a new name with the directory-output and required width/depth.

### Skip baking: use the official public dataset on HF

For automation, the simplest path is to point `--dataset` at the public HF dataset; no local bake step is needed.

```bash
uv run whest run --estimator estimator.py --runner subprocess --format json \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v1-warmup \
  --split mini
```

- The first invocation downloads about **2 GB** of parquet to `~/.cache/huggingface/`. Subsequent runs hit the cache. Account for the download time on a cold machine (CI runners, fresh containers).
- Splits on `v1-warmup`: **`mini`** (100 MLPs, the recommended default for most automation) and **`full`** (larger). `--split` is required when more than one split exists, which is the case here.
- `--n-mlps` is optional and clamps to the chosen split's size; omit it to score against all MLPs in the split, which is usually what you want.
- For a one-off run where you don't want to keep 2 GB on disk, add `--streaming` (iteration-only, no cache).

### Other commands (less load-bearing for automation)

- `uv run whest validate --estimator path/to/estimator.py` — checks the class resolves, `setup()` runs, `predict()` returns the right shape/dtype on a tiny MLP. Cheap; run before a long scoring pass.
- `uv run whest doctor` — environment health check (6 rows). Run when something's mysteriously broken.
- `uv run whest package --estimator path/to/estimator.py -o submission.tar.gz` — builds the AIcrowd submission tarball (Stage 6).

## The JSON schema (key fields)

`whest run --format json` returns a single JSON object. The fields you'll actually use:

```jsonc
{
  "schema_version": "1.1",
  "mode": "agent",
  "results": {
    "adjusted_final_layer_score": 0.0861,       // PRIMARY metric, lower is better
    "final_layer_mse": 0.8611,                  // raw MSE on the last layer
    "all_layers_mse": 0.6606,                   // raw MSE averaged across all layers
    "per_layer_mse": [0.317, 0.488, /* ... */, 0.861],  // length == depth
    "best_mlp_adjusted_final_layer_score": 0.0540,
    "worst_mlp_adjusted_final_layer_score": 0.1364,
    "mean_score_multiplier": 0.1,               // mean of max(0.1, eff_compute/budget)
    "mean_compute_utilization": 1.36e-05,       // mean of eff_compute/budget
    "mean_effective_compute": 921667,
    "n_failed_mlps": 0,
    "failure_breakdown": {
      "budget_exhausted": 0,
      "time_exhausted": 0,
      "residual_wall_time_exhausted": 0,
      "combined_budget_exhausted": 0,
      "error": 0
    },
    "per_mlp": [
      {
        "mlp_index": 0,                          // dataset-local index, NOT a grader seed
        "mlp_name": "christopher-calderon",      // stable human-readable id for the MLP
        "final_layer_mse": 0.8924,
        "all_layers_mse": 0.6592,
        "per_layer_mse": [/* depth values */],
        "adjusted_final_layer_score": 0.0892,
        "flops_used": 0,
        "effective_compute": 1766695,
        "wall_time_s": 3.5e-05,
        "flopscope_backend_time_s": 0.0,
        "flopscope_overhead_time_s": 1.7e-05,
        "residual_wall_time_s": 1.8e-05,
        "budget_exhausted": false,
        "time_exhausted": false,
        "residual_wall_time_exhausted": false,
        "combined_budget_exhausted": false,
        "traceback": null,                       // populated when this MLP errored
        "breakdowns": { "estimator": { /* flop_budget, flops_used/remaining, time fields, by_namespace */ } }
      }
      // ...one entry per MLP in the dataset
    ],
    "breakdowns": {
      "sampling":  { /* ground-truth costs, irrelevant for scoring */ },
      "estimator": { /* aggregate per-namespace flop & time across all MLPs */ }
    }
  },
  "run_config": {
    "n_mlps": 10,
    "width": 256,
    "depth": 8,
    "seed": null,                                // top-level run seed; null if --seed not passed
    "flop_budget": 68000000000,                  // 6.8e10
    "wall_time_limit_s": 60.0,
    "residual_wall_time_limit_s": null,
    "dataset": {
      "path": "/abs/path/to/dataset_dir",        // baked directory, NOT an .npz
      "sha256": "7c20...",                       // tag experiments with this for reproducibility
      "seed": null,                              // null when seeds were auto-generated; populated if you used --mlp-seeds
      "n_mlps": 10
    }
  },
  "run_meta": {
    "run_started_at_utc": "...",
    "run_finished_at_utc": "...",
    "run_duration_s": 3.96,
    "host": { /* hostname, os, python_version, numpy_version, flopscope_version, cpu/ram, ... */ }
  }
}
```

### Score formula

Per-MLP score: `final_layer_mse × max(0.1, effective_compute / flop_budget)`

Aggregate: arithmetic mean of per-MLP scores. Lower is better. The `max(0.1, ...)` floor means an estimator that returns garbage and uses zero compute still gets penalized by `0.1 × final_layer_mse` (cannot escape by skipping work).

### Reference numbers on the default suite

For `width=256, depth=8, flop_budget=6.8e10`:

| estimator | typical `final_layer_mse` | typical `adjusted_final_layer_score` |
|---|---|---|
| All-zeros baseline ([estimator.py](../../../estimator.py)) | ~0.81 | ~0.08 |
| `examples/02_mean_propagation.py` | ~5e-4 | ~5e-5 |
| `examples/03_covariance_propagation.py` | ~2.4e-5 | ~2.4e-6 |
| `examples/04_combined.py` | ~2.4e-5 | ~2.4e-6 |

The `adjusted_final_layer_score` column applies the `0.1` floor on the score multiplier (these examples all use far less than 10% of budget, so the floor is what actually multiplies their MSE). The MSE numbers come from [examples/README.md](../../../examples/README.md).

A "good idea" usually moves the score by an order of magnitude, not a few percent.

## Pitfalls to avoid

- **Do not parse the non-JSON Rich-formatted output.** It is decorated, line-wrapped, and not stable across versions. Always pass `--format json`.
- **`mlp_index` is not a grader seed.** It is 0..N-1 within the dataset. Two different datasets with the same `mlp_index` evaluate different MLPs. Always combine `mlp_index` with `run_config.dataset.sha256` when keying.
- **Without `--dataset`, every run pays the Monte Carlo cost.** A 10-MLP score takes ~30s without a dataset, ~3s with one. In loops this is the difference between minutes and hours.
- **MLP identity comes from the dataset, not `--n-mlps`/`--seed`.** With `--dataset`, MLP seeds are baked in: `--n-mlps` only clamps the suite size (and defaults to the full dataset), and `--seed` only reseeds the estimator's `setup()` call. To change which MLPs you score against, bake a new dataset.
- **`run_started_at_utc` and `run_finished_at_utc` differ by microseconds at most** in current `whestbench` (they timestamp the report-emission window, not the run). Trust `run_duration_s` instead.

## Minimal automation pattern

```python
import json, subprocess
from pathlib import Path

# dataset_path is a baked dataset directory (output of `whest dataset bake --output ...`),
# or an "hf://owner/repo[@rev]" string for HF Hub datasets.
result = subprocess.run(
    [
        "uv", "run", "whest", "run",
        "--estimator", str(estimator_path),
        "--runner", "subprocess",
        "--format", "json",
        "--dataset", str(dataset_path),
    ],
    capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent,
)
report = json.loads(result.stdout)
score = report["results"]["adjusted_final_layer_score"]
per_mlp = report["results"]["per_mlp"]
dataset_sha = report["run_config"]["dataset"]["sha256"]
```

Estimator failures (raised exceptions, budget exhaustion) show up as `n_failed_mlps > 0` and entries in `failure_breakdown`; the run itself still exits 0. Check those before trusting the score.

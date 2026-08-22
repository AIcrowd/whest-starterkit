# Profile Simulation Performance

> [← Documentation](../README.md)

## 🎯 When to use this page

The starter kit uses flopscope as its single simulation backend. The `profile-simulation` command lets you verify the backend computes correct values and see how its wall-clock cost scales with network size.

> This page covers the `whest profile-simulation` command for benchmarking the backend's correctness and timing. For managing your estimator's FLOP budget during development, see [Manage Your FLOP Budget](../how-to/manage-flop-budget.md).

Use this page when you want to:

- **Verify flopscope is installed and correct** — the profiler runs a pre-flight correctness check before timing anything.
- **Triage performance problems** (e.g. "fast on my laptop, slow in CI") — compare median wall-clock time for `run_mlp` and `sample_layer_statistics` across widths, depths, and sample counts.
- **Collect reproducible timing data** — JSON output includes correctness results and per-configuration timing (median time, raw per-iteration times, warmup time) across network sizes, plus hardware/library-version metadata.

## 🚀 Do this now

### 1. Run a quick profile

```bash
uv run whest profile-simulation --preset quick
```

This finishes in seconds and gives you a first look at correctness and timing across a couple of network sizes.

### 2. Run the standard profile

```bash
uv run whest profile-simulation
```

The default `standard` preset tests two widths (64, 256) and three depths (4, 32, 128). It gives a reliable picture of how run time scales across network sizes.

### 3. Save results for comparison

```bash
uv run whest profile-simulation --output results.json
```

The JSON file contains correctness results and per-configuration timing data (median time, raw per-iteration times, warmup time) across all tested configurations, plus hardware and library-version metadata.

## Choosing presets

| Preset | Widths | Depths | N_Samples | Typical time |
|--------|--------|--------|-----------|--------------|
| `super-quick` | 256 | 4 | 10 000 | Sub-second |
| `quick` | 256 | 4, 128 | 10 000, 100 000 | Seconds |
| `standard` | 64, 256 | 4, 32, 128 | 10 000, 100 000 | Under a minute |
| `exhaustive` | 64, 256 | 4, 32, 128 | 10 000, 100 000, 1 000 000 | Minutes |

Use `quick` for a fast sanity check and `standard` for development decisions.

## Understanding the output

The terminal table shows:

- **Hardware** — platform, CPU, RAM, and Python/NumPy versions, for reproducibility across machines.
- **Correctness** — PASS or FAIL for the flopscope backend. A FAIL indicates a version mismatch or installation problem.
- **Detail** — one row per (backend, width, depth, n_samples) combination, with median wall-clock time for `run_mlp` and `sample_layer_statistics`.

Add `--verbose` for the full **Timing Results** table: one row per (backend, operation, width, depth, n_samples) combination, with columns for median time, speedup vs NumPy, and status. It renders only in rich output, so add `--format rich` when piping or redirecting: `uv run whest profile-simulation --verbose --format rich | tee profile.txt`. For machine-readable data prefer `--output results.json`; the JSON carries the full per-configuration detail regardless of `--verbose`.

## Common workflows

### Debug a correctness failure

If the correctness check shows FAIL:

```bash
uv run whest profile-simulation --preset quick --debug
```

The error message will indicate whether the issue is a numerical tolerance failure or a missing dependency.

### Export timing data across machines or code changes

```bash
uv run whest profile-simulation --preset exhaustive --output timing_data.json
```

Compare the `timing` array's `median_time` field between two JSON files to see whether a code change, dependency bump, or a different machine affected simulation performance.

This command reports wall-clock timing only, not FLOP counts. To see how many FLOPs your *estimator* consumes, wrap it in a `flopscope.BudgetContext` and call `ctx.summary()`; see [Manage Your FLOP Budget](../how-to/manage-flop-budget.md).

## ✅ Expected outcome

- The terminal displays a formatted table with correctness and timing results.
- If `--output` is provided, a JSON file is written with hardware info, correctness results, and per-configuration timing data.

## ➡️ Next step

- [CLI Reference](../reference/cli-reference.md) — points to the upstream `whest` CLI reference for the full flag list
- [Use Evaluation Datasets](../how-to/use-evaluation-datasets.md) — pre-create datasets for faster iteration
- [Validate, Run, and Package](../how-to/validate-run-package.md) — score your estimator

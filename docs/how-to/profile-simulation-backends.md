# Profile Simulation Backends

## When to use this page

The starter kit uses mechestim as its single simulation backend — analytical FLOP counting replaces wall-clock timing across multiple backends. The `profile-simulation` command lets you verify FLOP accounting correctness and explore how FLOP costs scale with network size.

For a practical guide to managing your FLOP budget during development, see [Manage Your FLOP Budget](./manage-flop-budget.md).

Use this page when you want to:

- **Verify mechestim is installed and correct** — the profiler runs a pre-flight correctness check before reporting FLOP data.
- **Understand FLOP scaling** — see how FLOP costs grow with width, depth, and budget so you can calibrate your estimator's budget usage.
- **Collect reproducible profiling data** — JSON output includes correctness results and FLOP accounting across network sizes.

## Do this now

### 1. Run a quick profile

```bash
nestim profile-simulation --preset quick
```

This finishes in seconds and gives you a first look at FLOP accounting correctness and scaling.

### 2. Run the standard profile

```bash
nestim profile-simulation
```

The default `standard` preset tests two widths (64, 256) and five depths (4–128). It gives a reliable picture of FLOP cost scaling across network sizes.

### 3. Save results for comparison

```bash
nestim profile-simulation --output results.json
```

The JSON file contains correctness results and FLOP accounting data across all tested configurations.

## Choosing presets

| Preset | Widths | Depths | Typical time |
|--------|--------|--------|--------------|
| `super-quick` | 256 | 4 | Sub-second |
| `quick` | 256 | 4, 128 | Seconds |
| `standard` | 64, 256 | 4, 32, 128 | Under a minute |
| `exhaustive` | 64, 256 | 4, 32, 128 | Minutes |

Use `quick` for a fast sanity check and `standard` for development decisions.

## Understanding the output

The terminal table shows:

- **Pre-flight Correctness Check** — PASS or FAIL for the mechestim backend. A FAIL indicates a version mismatch or installation problem.
- **FLOP Accounting Results** — one row per (width, depth) combination showing the total FLOPs reported by mechestim for a forward pass.

## Common workflows

### Debug a correctness failure

If the correctness check shows FAIL:

```bash
nestim profile-simulation --preset quick --debug
```

The error message will indicate whether the issue is a numerical tolerance failure or a missing dependency.

### Export FLOP data for estimator calibration

```bash
nestim profile-simulation --preset exhaustive --output flop_data.json
```

Use the FLOP counts in the JSON to understand how much budget your estimator consumes per layer, and tune your budget-allocation logic accordingly.

## Expected outcome

- The terminal displays a formatted table with correctness and FLOP accounting results.
- If `--output` is provided, a JSON file is written with full profiling data.

## Next step

- [CLI Reference: profile-simulation](../reference/cli-reference.md#nestim-profile-simulation) — full flag reference
- [Use Evaluation Datasets](./use-evaluation-datasets.md) — pre-create datasets for faster iteration
- [Validate, Run, and Package](./validate-run-package.md) — score your estimator

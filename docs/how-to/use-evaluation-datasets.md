# Use Evaluation Datasets

## When to use this page

Every `nestim run` generates fresh random MLPs and samples thousands of forward passes to establish ground truth. This is correct but slow — especially when you are iterating on an estimator and re-running the same evaluation dozens of times during development.

Pre-created evaluation datasets let you do that expensive work once and reuse it across your entire development cycle:

- **Faster iteration** — `nestim run --dataset` skips MLP generation and ground truth sampling entirely.
- **Fair comparisons** — every estimator you test is scored against the exact same MLPs with the same ground truth.
- **Reproducibility** — the dataset file records the seed and all creation parameters, so anyone can recreate it exactly.

## Do this now

### 1. Create your dataset (once)

```bash
nestim create-dataset -o my_dataset.npz
```

This generates MLPs, samples ground truth means, and benchmarks sampling baselines. Everything is saved to a single `.npz` file.

Common options:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-mlps` | 10 | Number of random MLPs to generate |
| `--n-samples` | 10000 | Samples per MLP for ground truth estimation |
| `--seed` | auto | RNG seed (auto-generated if omitted, always recorded) |
| `-o, --output` | `eval_dataset.npz` | Output file path |
| `--width` | 100 | Neuron count per MLP |
| `--max-depth` | 30 | Layers per MLP |
| `--budgets` | `10,100,1000,10000` | Comma-separated budget list |

### 2. Run against it (every time)

```bash
nestim run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz
```

The `--n-mlps` and `--n-samples` flags are ignored when `--dataset` is provided — the values come from the dataset file.

You can keep reusing the same dataset file across your entire development cycle. Edit your estimator, re-run the command, compare scores — the ground truth stays the same so differences reflect only your estimator changes.

## ✅ Expected outcome

- `create-dataset` produces a `.npz` file at the specified path.
- `run --dataset` skips the "Sampling (Ground Truth)" progress bar.
- Score reports are consistent across runs with the same dataset.

## Understanding baseline times and hardware

The dataset file also stores *sampling baseline times* — measurements of how long a plain forward pass takes for each budget. These baselines are used during scoring to determine whether your estimator is faster or slower than naive sampling, and they directly affect your final score through time-ratio adjustments.

**Baselines are hardware-dependent.** A forward pass on a fast workstation takes less time than the same pass on a laptop. If you create a dataset on one machine and run it on another, the stored baselines would not reflect the actual performance characteristics of the current hardware, leading to incorrect time-ratio scoring.

### Default behavior: auto-recompute

When `nestim run --dataset` detects that the current machine differs from the one that created the dataset, it automatically recomputes the baselines:

```text
⚠ Dataset baselines computed on workstation-01 (Apple M2 Max, 12 cores, 32GB).
  Current hardware differs. Recomputing baselines...
```

The MLPs and ground truth are still loaded from the file — only the baselines are measured fresh. This adds a small one-time cost but ensures scores are accurate for your hardware.

### Strict mode

If you want to ensure that no one accidentally uses stale baselines (for example, in a CI pipeline or a formal evaluation), pass `--strict-baselines`:

```bash
nestim run --estimator ./estimator.py --dataset my_dataset.npz --strict-baselines
```

With `--strict-baselines`, the CLI refuses to run if the current hardware does not match the dataset's recorded hardware. This is useful when you need guaranteed consistency — for example, if the same machine that created the dataset must also run all evaluations.

## Dataset traceability

When using `--dataset`, the results JSON includes a `dataset` reference under `run_config` so you can always trace exactly which dataset produced a given score:

```json
{
  "run_config": {
    "dataset": {
      "path": "/path/to/my_dataset.npz",
      "sha256": "a1b2c3...",
      "seed": 42,
      "n_mlps": 10,
      "n_samples": 10000,
      "baselines_recomputed": false
    }
  }
}
```

## ➡️ Next step

- [Validate, Run, and Package](./validate-run-package.md)
- [Score Report Fields](../reference/score-report-fields.md)

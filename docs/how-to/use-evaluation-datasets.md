# Use Evaluation Datasets

## When to use this page

Every `whest run` generates fresh random MLPs and samples many forward passes to establish ground truth. This is correct but slow — especially when you are iterating on an estimator and re-running the same evaluation dozens of times during development.

Pre-created evaluation datasets let you do that expensive work once and reuse it across your entire development cycle:

- **Faster iteration** — `whest run --dataset` skips MLP generation and ground truth sampling entirely.
- **Fair comparisons** — every estimator you test is scored against the exact same MLPs with the same ground truth.
- **Reproducibility** — the dataset file records the seed and all creation parameters, so anyone can recreate it exactly.

## Do this now

### 1. Create your dataset (once)

```bash
whest create-dataset -o my_dataset.npz
```

This generates MLPs and samples ground truth means. Everything is saved to a single `.npz` file.

Common options:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-mlps` | 10 | Number of random MLPs to generate |
| `--n-samples` | 10000 | Samples per MLP for ground truth estimation |
| `--seed` | auto | RNG seed (auto-generated if omitted, always recorded) |
| `-o, --output` | `eval_dataset.npz` | Output file path |
| `--width` | (contest default) | Neuron count per MLP |
| `--depth` | (contest default) | Layers per MLP |
| `--flop-budget` | (contest default) | FLOP cap for the estimator |

### 2. Run against it (every time)

```bash
whest run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz
```

The `--n-mlps` flag is ignored when `--dataset` is provided — the values come from the dataset file.

You can keep reusing the same dataset file across your entire development cycle. Edit your estimator, re-run the command, compare scores — the ground truth stays the same so differences reflect only your estimator changes.

## Expected outcome

- `create-dataset` produces a `.npz` file at the specified path.
- `run --dataset` shows "Loading dataset" instead of "Generating MLPs" and skips ground truth sampling.
- Score reports are consistent across runs with the same dataset.

## Dataset portability

Unlike the old time-based scoring model, whest uses analytical FLOP counting rather than wall-clock timing. This means datasets are **fully portable across machines** — the stored ground truth and FLOP budgets are hardware-independent. You can create a dataset on a laptop and run it on a cloud instance with identical results.

## Dataset traceability

When using `--dataset`, the results JSON includes a `dataset` reference under `run_config` so you can always trace exactly which dataset produced a given score:

```json
{
  "run_config": {
    "dataset": {
      "path": "/path/to/my_dataset.npz",
      "sha256": "a1b2c3...",
      "seed": 42,
      "n_mlps": 10
    }
  }
}
```

## Next step

- [Validate, Run, and Package](./validate-run-package.md)
- [Score Report Fields](../reference/score-report-fields.md)

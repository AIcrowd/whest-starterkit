# Use Evaluation Datasets

> [← Documentation](../README.md)

## 🎯 When to use this page

Every `whest run` generates fresh random MLPs and samples many forward passes to establish ground truth. This is correct but slow — especially when you are iterating on an estimator and re-running the same evaluation dozens of times during development.

Pre-created evaluation datasets let you do that expensive work once and reuse it across your entire development cycle:

- **Faster iteration** — `whest run --dataset` skips MLP generation and ground truth sampling entirely.
- **Fair comparisons** — every estimator you test is scored against the exact same MLPs with the same ground truth.
- **Reproducibility** — the dataset artifact records the seed and all creation parameters, so anyone can recreate it exactly.

For reproducible datasets, pass an explicit JSON array of per-MLP seeds via `whest dataset bake --mlp-seeds path/to/seeds.json` (each seed a non-negative 63-bit int). When omitted, seeds are auto-generated with `secrets.randbits(63)` and stored in the dataset metadata so the result is still recoverable from the baked directory.

## 💾 Public starter dataset (recommended)

The public baseline dataset lives on Hugging Face: <https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026>.

Point `--dataset` at it directly to skip local baking. The first run downloads about **2 GB** of parquet shards and caches them under your standard HF cache (`~/.cache/huggingface/`); every subsequent run hits the cache.

```bash
whest run --estimator estimator.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v1-warmup \
  --split mini
```

The tag `v1-warmup` is the recommended pin for current work. Two splits are available:

- `--split mini`: **100 MLPs**. Good default for everyday iteration. Use this unless you have a reason not to.
- `--split full`: the larger sibling, for stronger statistical signal at the cost of longer evaluation.

`--n-mlps` is **optional** when `--dataset` is set. Omitted, it scores against every MLP in the chosen split (which is what you usually want). Pass a smaller number, for example `--n-mlps 10`, to clamp the suite for a quick sanity check; the dataset still supplies the per-MLP seeds for that subset.

If you want to stream the dataset rather than download it (handy for one-off CI runs that won't reuse it), add `--streaming`. Streaming is iteration-only and does not cache, so subsequent runs will re-fetch.

## 🚀 Do this now

### 1. Bake your dataset (once)

```bash
# Recommended local eval set. Low score noise, bakes in ~30s on a laptop.
whest dataset bake --n-mlps 30 --n-samples 100000 --width 256 --depth 8 --output my_dataset/

# Quick smoke-test set. Noisier scores but bakes in ~2s; use for one-shot sanity checks.
whest dataset bake --n-mlps 5 --n-samples 5000 --width 256 --depth 8 --output smoke_dataset/
```

For real comparisons between estimator ideas, prefer **30 MLPs with `--n-samples 100000`**. Fewer MLPs and fewer samples both add noise that can swamp the difference between good ideas.

> **Leaderboard sizing**: the AIcrowd grader scores against **50 MLPs with `--n-samples 1_000_000_000`** (10^9). Local scores track but do not exactly match the leaderboard at lower sample counts.

This generates MLPs and samples ground truth means. Output is a Hugging Face-compatible dataset **directory** (parquet + `metadata.json`) for easy sharing or upload to the Hub.

Required and common flags:

| Flag | Required? | Description |
|------|-----------|-------------|
| `--n-mlps N` | yes | Number of random MLPs to generate (30 recommended for real comparisons) |
| `--n-samples N` | yes | Samples per MLP for ground truth estimation (100000 recommended; 5000 for quick smoke checks) |
| `--width N` | yes | Neuron count per MLP (contest uses 256) |
| `--depth N` | yes | Layers per MLP (contest uses 8) |
| `--output DIR/` | yes | Output **directory** (must not already exist) |
| `--mlp-seeds FILE` | no | JSON array of N explicit per-MLP seeds; omit to auto-generate |
| `--split NAME` | no | Split name (defaults to `public`). Must match `[a-z][a-z0-9-]*` |
| `--torch` | no | Use GPU/torch backend instead of NumPy |
| `--mlps-per-batch N` | no | Memory/throughput knob for the GPU backend |

To publish the result: `whest dataset upload my_dataset/ --repo aicrowd/<name> --tag v1`.

If you want to avoid extra host probing during local development, set `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1` before `whest dataset bake` or `whest run`. This skips only the OS-native fallback probes used to fill missing hardware fields in report and dataset metadata. Cheap fields and `psutil`-backed fields are still recorded, and fallback-backed fields may remain `null`.

### 2. Run against it (every time)

```bash
whest run --estimator estimator.py --dataset my_dataset/
```

`--dataset` accepts a local directory (the output of `dataset bake`) or `hf://owner/repo[@revision]` for a Hub-hosted dataset.

With `--dataset` set, MLP seeds are baked in. `--n-mlps` defaults to the full dataset size and is clamped if you ask for more than the dataset contains. `--seed` only reseeds the estimator's `setup()` call; the per-MLP seeds still come from the dataset.

You can keep reusing the same dataset across your entire development cycle. Edit your estimator, re-run the command, compare scores. The ground truth stays the same so differences reflect only your estimator changes.

## ✅ Expected outcome

- `dataset bake` produces a HF-compatible directory at the specified location (parquet shards in `data/` plus `metadata.json`).
- `run --dataset` shows "Loading dataset" instead of "Generating MLPs" and skips ground truth sampling.
- `run --dataset` still shows a `Sampling Budget Breakdown (Ground Truth)` section in human output, restored from the dataset metadata for exactly the MLPs used in that run.
- Score reports are consistent across runs with the same dataset.

## Dataset portability

Unlike the old time-based scoring model, flopscope uses analytical FLOP counting rather than wall-clock timing. This means datasets are **fully portable across machines** — the stored ground truth and FLOP budgets are hardware-independent. You can create a dataset on a laptop and run it on a cloud instance with identical results.

## Dataset traceability

When using `--dataset`, the results JSON includes a `dataset` reference under `run_config` so you can always trace exactly which dataset produced a given score:

```json
{
  "run_config": {
    "dataset": {
      "path": "/abs/path/to/my_dataset",
      "sha256": "a1b2c3...",
      "seed": null,
      "n_mlps": 10
    }
  }
}
```

`dataset.seed` is `null` when seeds were auto-generated; pass `--mlp-seeds` to `dataset bake` to make it deterministic across machines.

Example seeded run command:

```bash
whest run --estimator estimator.py --seed 20260417 --dataset my_dataset/
```

`run --seed` stores the chosen seed in `run_config.seed`, and the baked dataset's `metadata.json` records:

```json
{"seed_protocol": {"name": "whestbench_explicit_per_mlp_seeds", "version": "3.0"}}
```

## ➡️ Next step

- [Validate, Run, and Package](./validate-run-package.md)
- [Score Report Fields](../reference/score-report-fields.md)

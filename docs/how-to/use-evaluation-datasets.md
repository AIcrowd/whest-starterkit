# Use Evaluation Datasets

> [← Documentation](../README.md)

## 🎯 When to use this page

Every `whest run` *without* `--dataset` generates fresh random MLPs and runs millions of forward passes to establish ground-truth means. That's slow when you're iterating: you pay the ground-truth tax on every run, and you can't compare two estimator versions on identical MLPs.

Pre-baked evaluation datasets fix both:

- **Fast iteration** — ground truth is precomputed; `whest run --dataset ...` skips MLP generation and Monte-Carlo sampling entirely.
- **Fair comparisons** — every estimator you test scores against the exact same MLPs against the exact same ground-truth means.
- **Reproducibility** — the dataset's `metadata.json` pins the seeds, schema, and bake config, so anyone can verify your numbers.

For day-to-day estimator work, you almost never need to bake your own. The AIcrowd team publishes a pre-baked dataset on HuggingFace Hub; just point `whest run` at it.

## 🚀 Do this now (HF Hub, no bake required)

The published Public Release dataset is at [`aicrowd/arc-whestbench-public-2026`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026) and contains two splits:

| Split | Size | Use for |
|---|---:|---|
| `mini` | 100 MLPs (~250 MB) | Day-to-day iteration. Downloads in seconds. |
| `full` | 1,000 MLPs (~1.4 GB) | Final lock-in check before you submit. |

`mini` is the **default split** — `whest run --dataset hf://...` without `--split` picks it automatically.

### 1. Iterate against mini

```bash
whest run \
    --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v1-warmup
```

The CLI prints something like `Using default split 'mini' (from metadata.default_split)`, downloads ~250 MB on the first run (cached for every subsequent run), and runs your estimator against 100 MLPs. Typical end-to-end wall time after the cache is warm: under 5 seconds.

### 2. Lock in your numbers against full

```bash
whest run \
    --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v1-warmup \
    --split full
```

Use this before submitting. `mini` is independent of `full` (different MLPs entirely), so a good mini score doesn't guarantee a good full score — but big regressions on full almost always show up on mini first.

### 3. Same dataset via the pure HF API

If you want the raw rows for analysis (rather than running an estimator), use `datasets`:

```python
from datasets import load_dataset

# mini is the default config of this repo
mini = load_dataset("aicrowd/arc-whestbench-public-2026",
                    revision="v1-warmup", split="mini")
print(mini[0]["mlp_name"])     # e.g. "krista-wright"
print(mini[0]["weights"])      # (depth=8, width=256, width=256) float64

# full is a separate config; pass the config name explicitly
full = load_dataset("aicrowd/arc-whestbench-public-2026",
                    "full", revision="v1-warmup", split="full")
```

The dataset is stored on HF Hub via [Xet](https://huggingface.co/docs/hub/xet), so re-downloads dedupe at the chunk level and parallel multi-shard fetches are fast. For maximum download throughput on a fast connection, set `HF_XET_HIGH_PERFORMANCE=1` in your environment before the load.

> **Tip — prepared-Arrow fast path.** When you load via `whestbench.load_dataset(...)` (or via `whest run --dataset hf://...`), WhestBench prefers a pre-built `prepared/<split>/` Arrow artifact published alongside the parquet. It downloads only that subtree and memory-maps it via `datasets.Dataset.load_from_disk()`, skipping the parquet→arrow conversion that the bare `datasets.load_dataset(...)` path runs on first use. End-to-end this is ~18% faster on `mini` and ~60% faster on `full`, with a ~33% smaller cache footprint. You'll see a one-line stderr notice (`whestbench: using prepared Arrow split 'mini' from ...`) when the fast path fires. Falls back silently to the parquet path if anything goes wrong.

## 🛠 Bake your own (rare)

You only need this when:

- You're testing on MLPs the public dataset doesn't include (a different width / depth, or a private seed list).
- You want to validate a custom bake config end-to-end.

The modern command is `whest dataset bake`. It writes a *directory* (not a `.npz`) in the schema-3.0 layout used by HF Hub:

```bash
whest dataset bake \
    --output ./my-eval \
    --n-mlps 10 \
    --n-samples 1_000_000 \
    --width 256 \
    --depth 8
# Produces:
#   ./my-eval/
#   ├── data/public-00000-of-00001.parquet
#   ├── metadata.json
#   └── README.md
```

Common flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-mlps` | 10 | Number of MLPs to bake |
| `--n-samples` | 10000 | Ground-truth samples per MLP |
| `--width` | 256 | Neurons per layer |
| `--depth` | 8 | Number of weight matrices |
| `--output` / `-o` | (required) | Output directory (must not exist) |
| `--mlp-seeds` | auto | JSON file with per-MLP seeds; defaults to fresh `secrets.randbits(63)` |
| `--split` | `public` | Split name for the parquet file |

Then run against it like any HF dataset:

```bash
whest run --estimator estimator.py --dataset ./my-eval
```

If you want to avoid extra host probing during local bakes, set `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1` before `whest dataset bake` or `whest run`. This skips only the OS-native fallback probes used to fill missing hardware fields in metadata. Cheap fields and `psutil`-backed fields are still recorded.

## ✅ Expected outcome

- `whest run --dataset hf://...@v1-warmup` (no `--split`) auto-resolves to `mini`, downloads ~250 MB on first call, scores in seconds on subsequent calls.
- `whest run --dataset hf://...@v1-warmup --split full` deliberately switches to the 1,000-MLP split.
- Re-running with the same dataset + estimator gives identical scores (the bake is deterministic).

## 📚 Dataset traceability

When you use `--dataset`, the results JSON records exactly which dataset produced the score:

```json
{
  "run_config": {
    "dataset": {
      "path": "hf://aicrowd/arc-whestbench-public-2026@v1-warmup",
      "split": "mini",
      "n_mlps": 100
    }
  }
}
```

The dataset's own `metadata.json` pins `seed_protocol`, `whestbench_version`, `bake_config`, and the per-pod hardware fingerprints. Anyone can verify the bake from a commit OID + seed list.

## ➡️ Next step

- [Validate, Run, and Package](./validate-run-package.md)
- [Score Report Fields](../reference/score-report-fields.md)

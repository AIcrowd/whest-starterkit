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

The published Public Release dataset is at [`aicrowd/arc-whestbench-public-2026`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026). The Phase 1 MLPs are **256×32** (width 256, depth 32); the earlier `v1-warmup` round used 256×8. The dataset contains two splits:

| Split | Size | Use for |
|---|---:|---|
| `mini` | 100 MLPs (~850 MB) | Day-to-day iteration. Downloaded once, then served from cache. |
| `full` | 1,000 MLPs (~8.5 GB) | Final lock-in check before you submit. |

`mini` is the **default split** — `whest run --dataset hf://...` without `--split` picks it automatically.

> **Pin the `v1-phase1` revision.** Every command below pins `@v1-phase1`. Don't
> drop the tag and rely on bare `main`: `main` advances each contest phase, so an
> unpinned load can silently change the dataset underneath you — and an offline
> cache can just as silently keep serving an older phase. The tag is immutable
> and reproducible.

### 1. Iterate against mini

```bash
whest run \
    --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1
```

The CLI prints something like `Using default split 'mini' (from metadata.default_split)`, downloads ~850 MB on the first run (cached for every subsequent run), and runs your estimator against 100 MLPs. Subsequent runs reuse the cache, so there is no re-download.

### 2. Lock in your numbers against full

```bash
whest run \
    --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
    --split full
```

Use this before submitting. `mini` is independent of `full` (different MLPs entirely), so a good mini score doesn't guarantee a good full score — but big regressions on full almost always show up on mini first.

### 3. Same dataset via the pure HF API

If you want the raw rows for analysis (rather than running an estimator), use `datasets`:

```python
from datasets import load_dataset

# mini is the default config of this repo
mini = load_dataset("aicrowd/arc-whestbench-public-2026",
                    revision="v1-phase1", split="mini")
print(mini[0]["mlp_name"])     # e.g. "krista-wright"
print(mini[0]["weights"])      # (depth=32, width=256, width=256) float64  — warmup round was depth=8

# full is a separate config; pass the config name explicitly
full = load_dataset("aicrowd/arc-whestbench-public-2026",
                    "full", revision="v1-phase1", split="full")
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
    --depth 32
# Produces:
#   ./my-eval/
#   ├── data/public-00000-of-00001.parquet
#   ├── metadata.json
#   └── README.md
```

Common flags:

| Flag | Required / default | Description |
|------|--------------------|-------------|
| `--n-mlps` | **required** | Number of MLPs to bake |
| `--n-samples` | **required** | Ground-truth samples per MLP |
| `--width` | **required** | Neurons per layer |
| `--depth` | **required** | Number of weight matrices |
| `--output` | **required** | Output directory (must not exist) |
| `--mlp-seeds` | auto | JSON file with an array of per-MLP seeds (each `int < 2**63`); defaults to fresh `secrets.randbits(63)` |
| `--split` | `public` | Split name for the parquet file |
| `--config` | `default` | HF dataset config name for the split |

Then run against it like any HF dataset:

```bash
whest run --estimator estimator.py --dataset ./my-eval
```

If you want to avoid extra host probing during local bakes, set `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1` before `whest dataset bake` or `whest run`. This skips only the OS-native fallback probes used to fill missing hardware fields in metadata. Cheap fields and `psutil`-backed fields are still recorded.

## ⚡ Bake on a GPU (large datasets)

The default bake runs on CPU through flopscope. For large bakes — roughly `--n-samples ≥ 1e8`, where the CPU path gets slow — switch to the torch backend, which batches the forward passes on a GPU. It needs the optional `gpu` extra:

```bash
pip install 'whestbench[gpu]'      # pulls torch>=2.1
```

Then add `--torch` to engage it:

```bash
whest dataset bake \
    --torch --device cuda \
    --output ./my-big-eval \
    --n-mlps 1000 \
    --n-samples 1_000_000_000 \
    --width 256 \
    --depth 32
```

> ⚠️ **`--device` does nothing without `--torch`.** Running `whest dataset bake --device cuda` *without* `--torch` silently falls back to the slow CPU path and ignores your GPU. Always pass `--torch` for a GPU bake.

| Flag | Default | Description |
|------|---------|-------------|
| `--torch` | off | Use the GPU/torch backend. **Required** for any GPU bake. |
| `--device` | `auto` | `auto` \| `cuda` \| `mps` \| `cpu`. `auto` resolves cuda → mps → cpu. An explicit value errors if that device is unavailable (no silent CPU fallback). |
| `--mlps-per-batch` | `min(n_mlps, 16)` | MLPs processed in parallel on-device per batch. Lower it if you hit out-of-memory. |
| `--chunk-size` | auto | Samples per on-device chunk. CUDA: memory-aware (~25% of free VRAM, clamped 65 536–1 048 576); MPS/CPU: 65 536. |

Leaving `--mlps-per-batch` and `--chunk-size` unset lets the backend auto-tune to your VRAM, which is usually what you want. The output directory layout is identical to a CPU bake; its `metadata.json` records `backend: "torch"`, the resolved `device`, `torch_version`, and a `bake_config` determinism block.

The torch path is statistically — not bit-for-bit — equivalent to the CPU path at the same seeds (per-neuron means agree within ~3e-5 at N=1e9). For bit-exact reproducibility on CUDA you must additionally set `torch.backends.cudnn.deterministic = True`; the bake records the determinism state it ran under in `bake_config`.

### Shard a large bake across GPUs or hosts

To split one logical dataset across several GPUs or machines, bake contiguous slices in parallel and merge them. Pin a **shared seed list** first so every shard belongs to the same logical dataset:

```bash
# One shared seed file for all shards
python -c "import json, secrets; json.dump([secrets.randbits(63) for _ in range(1000)], open('seeds.json', 'w'))"

# Each GPU/host bakes one slice (0-indexed). Run these concurrently.
whest dataset bake --torch --device cuda --mlp-seeds seeds.json \
    --n-mlps 1000 --n-samples 1_000_000_000 --width 256 --depth 32 \
    --slice 0/4 --output ./shard-0
# ...repeat with --slice 1/4, 2/4, 3/4 on the other devices...

# Recombine the partial bakes into one dataset
whest dataset merge ./shard-0 ./shard-1 ./shard-2 ./shard-3 --output ./my-big-eval
```

Each shard writes a *partial* dataset (its `metadata.json` carries `is_partial: true`, `mlp_range`, and `total_n_mlps`); `whest dataset merge` stitches the partials back into one complete dataset. Prefer `--mlp-range START-END` (inclusive on both ends) over `--slice K/N` if you'd rather address explicit MLP ranges. The shared `--mlp-seeds` file is what keeps per-MLP identities and names stable across shards — don't let each shard roll its own seeds.

## ✅ Expected outcome

- `whest run --dataset hf://...@v1-phase1` (no `--split`) auto-resolves to `mini`, downloads ~850 MB on first call, then runs from cache on subsequent calls.
- `whest run --dataset hf://...@v1-phase1 --split full` deliberately switches to the 1,000-MLP split.
- Re-running with the same dataset + estimator gives identical scores (the bake is deterministic).

## 📚 Dataset traceability

When you use `--dataset`, the results JSON records exactly which dataset produced the score:

```json
{
  "run_config": {
    "dataset": {
      "path": "hf://aicrowd/arc-whestbench-public-2026@v1-phase1",
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

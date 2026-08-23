# Use evaluation datasets

> [← Documentation](../README.md)

## 🎯 When to use this page

Every `whest run` *without* `--dataset` generates fresh random MLPs and runs millions of forward passes to establish ground-truth means. That's slow when you're iterating: you regenerate the ground truth on every run, and you can't compare two estimator versions on identical MLPs.

Pre-baked evaluation datasets fix both, and add something the generated path cannot:

- **Fast iteration** — ground truth is precomputed; `whest run --dataset ...` skips MLP generation and Monte-Carlo sampling entirely.
- **Fair comparisons** — every estimator you test scores against the exact same MLPs against the exact same ground-truth means.
- **Reproducibility** — the dataset's `metadata.json` pins the schema, the seed protocol, and the bake provenance, so anyone can verify your numbers.

For day-to-day estimator work, you almost never need to bake your own. The AIcrowd team publishes a pre-baked dataset on HuggingFace Hub; point `whest run` at it.

## 🚀 Do this now (HF Hub, no bake required)

The published Public Release dataset is at [`aicrowd/arc-whestbench-public-2026`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026). The Phase 2 MLPs are **1024×16** (width 1024, depth 16); Phase 1 used 256×32, and the earlier `v1-warmup` round 256×8. The dataset contains two splits:

| Split | Contents | Use for |
|---|---:|---|
| `mini` | 100 MLPs | Day-to-day iteration. Downloaded once, then served from cache. |
| `full` | 1,000 MLPs | Final lock-in check before you submit. |

`mini` is the **default split**: `whest run --dataset hf://...` without `--split` picks it automatically.

> **Budget the download. Phase 2 is 8× Phase 1.** One MLP's weights are
> 16 × 1024 × 1024 float32 (64 MiB) against Phase 1's 32 × 256 × 256 (8 MiB), so
> every split grew with it:
>
> | Split | MLPs | Phase 1 (`@v1-phase1`) | Phase 2 (`@v2-phase2`) |
> |---|---:|---:|---:|
> | `mini` | 100 | 0.86 GB | **7.03 GB** |
> | `full` | 1000 | 8.59 GB | **69.96 GB** |
>
> Downloads are cached after the first call, so the cost is one-time per
> revision. But `full` is a 70 GB commitment. Start with `mini`, which is the default
> split and is what the Stage 3 walkthrough uses. `whest run` without `--dataset`
> needs no download at all and generates MLPs locally.

> **Pin the `v2-phase2` revision.** Every command below pins `@v2-phase2`. Don't
> drop the tag and rely on bare `main`: `main` advances each contest phase, so an
> unpinned load can silently change which dataset you get, and an offline
> cache can silently keep serving an older phase. The tag is immutable
> and reproducible.
>
> Each tag pins a whole round's configuration, not only one bake: `@v1-phase1`
> is 256×32 at a `2.72e11` budget with residual time *included in the score*,
> and `@v2-phase2` is 1024×16 at `2**41` with residual time *capped*.
> [Competition Rounds](../reference/rounds.md)
> lays every round out side by side, including which flags you have to restore to
> reproduce an old score, and why a number from before Phase 2 does not compare.

### 1. Iterate against mini

```bash
uv run whest run \
    --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2
```

The CLI prints something like `Using default split 'mini' (from metadata.default_split)`, downloads the split on the first run (cached for every subsequent run), and runs your estimator against 100 MLPs. Subsequent runs reuse the cache, so there is no re-download.

### 2. Lock in your numbers against full

```bash
uv run whest run \
    --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 \
    --split full
```

Use this before submitting. `mini` is independent of `full` (different MLPs entirely), so a good mini score doesn't guarantee a good full score, but big regressions on full almost always show up on mini first.

### 3. Same dataset via the pure HF API

If you want the raw rows for analysis (rather than running an estimator), use `datasets`:

```python
from datasets import load_dataset

# mini is the default config of this repo
mini = load_dataset("aicrowd/arc-whestbench-public-2026",
                    revision="v2-phase2", split="mini")
mini = mini.with_format("numpy")
print(mini[0]["mlp_name"])                                    # e.g. "kathleen-mueller"
print(mini[0]["weights"].shape, mini[0]["weights"].dtype)     # (16, 1024, 1024) float32   — phase 1 was (32, 256, 256)

# full is a separate config; pass the config name explicitly
full = load_dataset("aicrowd/arc-whestbench-public-2026",
                    "full", revision="v2-phase2", split="full")
```

Weights are **float32** as stored (the stored feature is `Array3D(shape=(16, 1024, 1024), dtype='float32')`), and `whestbench.load_dataset` / `whest run` keep them float32. Without `.with_format("numpy")` a row's `weights` is a plain Python list; `np.asarray(...)` on it silently gives you float64, which flopscope bills at 2x.

The dataset is stored on HF Hub via [Xet](https://huggingface.co/docs/hub/xet), so re-downloads dedupe at the chunk level and parallel multi-shard fetches are fast. For maximum download throughput on a fast connection, set `HF_XET_HIGH_PERFORMANCE=1` in your environment before the load.

> **Prepared-Arrow fast path (not published for `v2-phase2`).** When a dataset's
> `metadata.json` declares a `prepared_splits` block, `whestbench.load_dataset`
> downloads only the `prepared/<split>/` Arrow subtree and memory-maps it with
> `datasets.Dataset.load_from_disk()`, skipping the parquet→arrow conversion that
> the bare `datasets.load_dataset(...)` path runs on first use. The `v1-phase1`
> and `v1-warmup` revisions ship one; `@v2-phase2` does not, so every Phase 2
> load today takes the parquet path. Nothing to configure either way; it falls
> back silently.

> **Prefetch without a local copy.** With no `--output`, `whest dataset download`
> fetches into the Hugging Face hub cache only. A later
> `whest run --dataset hf://…` is then a pure cache hit. Pass `--output DIR`
> when you additionally want the files materialised into a directory.
>
> ```bash
> uv run whest dataset download aicrowd/arc-whestbench-public-2026 \
>     --revision v2-phase2 --split mini
> ```
>
> **Pass `--split`.** Without it this fetches the whole repo at that revision:
> both splits, ~77 GB (7.03 GB `mini` + 69.96 GB `full`). `--split mini` gets the
> 7 GB one plus `metadata.json` / `README.md`; pass `--split full` deliberately
> when you actually want the 70 GB one.

## 🛠 Bake your own (rare)

You only need this when:

- You're testing on MLPs the public dataset doesn't include (a different width / depth, or a private seed list).
- You want to validate a custom bake config end-to-end.

The modern command is `whest dataset bake`. It writes a *directory* (not a `.npz`) in the schema-3.0 layout used by HF Hub:

```bash
uv run whest dataset bake \
    --output ./my-eval \
    --n-mlps 10 \
    --n-samples 100_000 \
    --width 1024 \
    --depth 16
# ~80 s and ~670 MB for 10 MLPs at 1024x16 on a laptop; sampling time scales
# linearly in --n-samples

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
uv run whest run --estimator estimator.py --dataset ./my-eval
```

If you want to avoid extra host probing during local bakes, set `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1` before `whest dataset bake` or `whest run`. This skips only the OS-native fallback probes used to fill missing hardware fields in metadata. Cheap fields and `psutil`-backed fields are still recorded.

## ⚡ Bake on a GPU (large datasets)

The default bake runs on CPU through flopscope. For large bakes (roughly `--n-samples ≥ 1e8`, where the CPU path gets slow), switch to the torch backend, which batches the forward passes on a GPU. It needs the optional `gpu` extra:

```bash
uv add 'whestbench[gpu]'      # pulls torch>=2.1 into the project venv
# or, for a throwaway install without touching pyproject.toml:
#   uv pip install 'whestbench[gpu]'
```

To engage it, add `--torch`:

```bash
uv run whest dataset bake \
    --torch --device cuda \
    --output ./my-big-eval \
    --n-mlps 1000 \
    --n-samples 1_000_000_000 \
    --width 1024 \
    --depth 16
```

> ⚠️ **`--device` does nothing without `--torch`.** Running `whest dataset bake --device cuda` *without* `--torch` silently falls back to the slow CPU path and ignores your GPU. For a GPU bake, always pass `--torch`.

| Flag | Default | Description |
|------|---------|-------------|
| `--torch` | off | Use the GPU/torch backend. **Required** for any GPU bake. |
| `--device` | `auto` | `auto` \| `cuda` \| `mps` \| `cpu`. `auto` resolves cuda → mps → cpu. An explicit value errors if that device is unavailable (no silent CPU fallback). |
| `--mlps-per-batch` | `min(n_mlps, 16)` | MLPs processed in parallel on-device per batch. Lower it if you hit out-of-memory. |
| `--chunk-size` | auto | Samples per on-device chunk. CUDA: memory-aware (~25% of free VRAM, clamped 65 536–1 048 576); MPS/CPU: 65 536. |
| `--compile` | off | CUDA only. Inductor-compiled + CUDA-graphed fused sampling kernel, ~1.85× faster at width=256. Means are bit-identical; `avg_variance` within ~1 fp64 ULP. Recorded in metadata as `torch_compile`; pin your torch version if the bake must be reproducible or sharded. |

Leaving `--mlps-per-batch` and `--chunk-size` unset lets the backend auto-tune to your VRAM, which is usually what you want. The output directory layout is identical to a CPU bake; its `metadata.json` records `backend: "torch"`, the resolved `device`, `torch_version`, and a `bake_config` determinism block.

The torch path is statistically (not bit-for-bit) equivalent to the CPU path at the same seeds (per-neuron means agree within ~3e-5 at N=1e9). For bit-exact reproducibility on CUDA, all four determinism levers must match across bakes (`torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`), and you must pin `--chunk-size` to the same value across every shard and re-bake, since float64 accumulation is not associative. The bake records all four in `bake_config` and the chunk size alongside it, on a fixed torch version and GPU architecture.

### Shard a large bake across GPUs or hosts

To split one logical dataset across several GPUs or machines, bake contiguous slices in parallel and merge them. Pin a **shared seed list** first so every shard belongs to the same logical dataset:

```bash
# One shared seed file for all shards
uv run python -c "import json, secrets; json.dump([secrets.randbits(63) for _ in range(1000)], open('seeds.json', 'w'))"

# Each GPU/host bakes one slice (0-indexed). Run these concurrently.
uv run whest dataset bake --torch --device cuda --mlp-seeds seeds.json \
    --n-mlps 1000 --n-samples 1_000_000_000 --width 1024 --depth 16 \
    --slice 0/4 --output ./shard-0
# ...repeat with --slice 1/4, 2/4, 3/4 on the other devices...

# Recombine the partial bakes into one dataset
uv run whest dataset merge ./shard-0 ./shard-1 ./shard-2 ./shard-3 --output ./my-big-eval
```

Each shard writes a *partial* dataset (its `metadata.json` carries `is_partial: true`, `mlp_range`, and `total_n_mlps`); `whest dataset merge` combines the partials back into one complete dataset. Prefer `--mlp-range START-END` (inclusive on both ends) over `--slice K/N` if you'd rather address explicit MLP ranges. The shared `--mlp-seeds` file is what keeps per-MLP identities and names stable across shards; don't let each shard generate its own seeds.

## ✅ Expected outcome

- `whest run --dataset hf://...@v2-phase2` (no `--split`) auto-resolves to `mini`, downloads it on first call, then runs from cache on subsequent calls.
- `whest run --dataset hf://...@v2-phase2 --split full` deliberately switches to the 1,000-MLP split.
- Re-running with the same dataset + estimator gives identical scores (the bake is deterministic).

## 📚 Dataset traceability

When you use `--dataset`, the results JSON records exactly which dataset produced the score:

```json
{
  "run_config": {
    "dataset": {
      "path": "hf://aicrowd/arc-whestbench-public-2026@v2-phase2",
      "sha256": "aab61cf11271d20118e2109065c165a931dfacb493849f0d611f7a2d82a03790",
      "seed": null,
      "n_mlps": 100
    }
  }
}
```

`sha256` is the hash of the dataset's `metadata.json`. That is the field that pins *which* bake you scored against. The split name is not recorded; infer it from `n_mlps` (100 = `mini`, 1000 = `full`).

The dataset's own `metadata.json` pins `schema_version`, `seed_protocol`, `width` / `depth` / `n_samples`, and a `hardware_fingerprints[]` array, one entry per baking pod, each carrying that pod's `whestbench_version`, `flopscope_version`, and (torch bakes only) its `bake_config` determinism block. The per-MLP seeds live in the parquet `mlp_seed` column, not in metadata. Together with the revision's commit OID that is enough to re-bake and diff.

## ➡️ Next step

- [Validate, Run, and Package](./validate-run-package.md)
- [Score Report Fields](../reference/score-report-fields.md)

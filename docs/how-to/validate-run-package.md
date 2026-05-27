# Validate, Run, and Package

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page for the standard local participant loop.

## 🚀 Do this now

Validate estimator loading and output contract:

> `whest validate` is a fast sanity check using a small fixed MLP (width=4, depth=2). It verifies loading, output shape, and value finiteness — not full behavioral or performance correctness. Always follow with `whest run` for realistic tests.

```bash
whest validate --estimator estimator.py
```

Run local scoring (recommended default runner):

```bash
whest run --estimator estimator.py
```

`whest run` defaults to `--runner local` for fast iteration.

Run against a pre-created dataset (skips sampling — much faster for repeated runs).

**Option A: use the official public dataset on Hugging Face** (recommended; no bake step required). First run downloads ~2 GB and caches it under `~/.cache/huggingface/`; subsequent runs are instant.

```bash
whest run --estimator estimator.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v1-warmup \
  --split mini
```

Splits: `mini` (100 MLPs, recommended default) or `full` (larger). `--n-mlps` is optional with `--dataset` and clamps to the split size if passed.

**Option B: bake your own dataset locally** (for ad-hoc experiments or held-out test sets).

```bash
# Recommended local eval set (30 MLPs, 10^5 samples). Bakes in ~30s.
whest dataset bake --n-mlps 30 --n-samples 100000 --width 256 --depth 8 --output my_dataset/

# Quick smoke-test set (5 MLPs, 5000 samples). Noisier scores but bakes in ~2s.
whest dataset bake --n-mlps 5 --n-samples 5000 --width 256 --depth 8 --output smoke_dataset/

whest run --estimator estimator.py --dataset my_dataset/
```

The AIcrowd leaderboard grades against **50 MLPs with `--n-samples 1_000_000_000`** (10^9). Local scores track but do not exactly match the leaderboard at lower sample counts. See [Use Evaluation Datasets](./use-evaluation-datasets.md) for details.

Run faster local debug path:

```bash
whest run --estimator estimator.py --runner local
```

Run with machine-readable output:

```bash
whest run --estimator estimator.py --runner local --format json
```

`--json` still works as an alias, but `--format rich|plain|json` is the canonical output selector across the CLI.

Package submission artifact:

```bash
whest package --estimator estimator.py --output ./submission.tar.gz
```

Optional files during packaging:

```bash
whest package \
  --estimator estimator.py \
  --requirements requirements.txt \
  --submission-metadata submission.yaml \
  --approach APPROACH.md \
  --output ./submission.tar.gz
```

## Useful `whest run` flags

These all show up in `whest run --help` but get lost there. Reach for them when:

| Flag | Reach for it when… |
|---|---|
| `--seed N` | Deterministic comparison between two estimator versions. Pin the seed and the same MLPs, the same per-MLP `mlp.seed` values, and the same `SetupContext.seed` are used across runs. Also accepted by `whest validate` (seeds the validation `setup(ctx)` call). With `--dataset`, the dataset supplies the per-MLP seeds and `--seed` controls `ctx.seed` only. |
| `--n-samples N` | Ground-truth sampling samples per MLP. The contest default (in `whest run` without an explicit override) is `width * width * 256` (= 2,560,000 at the contest width of 100). For `whest dataset bake`, `--n-samples` has no default and must be passed. Use `100000` for real comparisons; drop to `5000` for a fast smoke check. The leaderboard grader uses `1_000_000_000` (10^9). |
| `--n-mlps N` | Default `10`. Drop to `3` while iterating to halve runtime; raise to `20+` when you're trying to reduce noise on a close score. |
| `--flop-budget N` | Default `6.8e10` (the grader effective-compute budget — caps `C_m = F_m + λ·R_m`, not just analytical FLOPs). Bump to `1e11` to confirm an algorithm idea isn't budget-starved before optimizing for budget. |
| `--profile` | Emits a per-namespace FLOP/time breakdown so you can see where your estimator burns the budget. |
| `--show-diagnostic-plots` | Renders convergence and per-layer error plots inline (terminal-friendly). Pairs well with `--profile`. |
| `--max-threads N` | Pin the BLAS thread pool size so `wall_time_s` is comparable across machines. Useful when triaging a "fast on my laptop, slow in CI" report. |
| `--detail {raw,full}` | `raw` strips Rich formatting (handy for `tee`-ing logs); `full` adds the per-MLP raw arrays. |
| `--wall-time-limit S` | Cap each `predict()` call's wall time. Useful when local debugging hangs on a numerical edge case. |
| `--residual-wall-time-limit S` | Cap time spent outside flopscope ops (Python plumbing, scipy calls). Surfaces "looks fast in FLOPs but Python is the bottleneck" issues. |
| `--debug` + `--fail-fast` | First exception → halt + raw traceback. Combine for the fastest "what broke?" loop. |

## ✅ Expected outcome

- `validate` passes,
- `run` produces a score report,
- `package` creates a `.tar.gz` artifact.

## ⚠️ Common first failure

Symptom: `run` fails after `validate` passed.

Use this escalation flow:

1. Retry with debug info:

```bash
whest run --estimator estimator.py --debug
```

2. If traceback still feels opaque, rerun in-process:

```bash
whest run --estimator estimator.py --runner local --debug
```

For runner modes, see [Stage 3: Run Locally](../getting-started/stage-3-run-local.md), [Stage 4: Subprocess Runner](../getting-started/stage-4-run-subprocess.md), and the [Debugging Checklist](./debugging-checklist.md).

Concrete example:

```text
Error [predict:PREDICT_ERROR]: Estimator predict failed.
Use --debug to include a traceback.
Tip: For estimator-level tracebacks, rerun with --runner local --debug.
```

Next command to run:

```bash
whest run --estimator estimator.py --runner local --debug
```

## ➡️ Next step

- [Use Evaluation Datasets](./use-evaluation-datasets.md)
- [CLI Reference](../reference/cli-reference.md)
- [Score Report Fields](../reference/score-report-fields.md)

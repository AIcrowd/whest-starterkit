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

Run against the published evaluation dataset on HuggingFace (skips sampling — much faster for repeated runs, no local bake needed):

```bash
whest run \
    --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v1-warmup
# auto-resolves to the `mini` split (100 MLPs, ~250 MB cached after first call)
```

Or bake a custom local dataset once and reuse it:

```bash
whest dataset bake --output ./my-eval --n-mlps 10 --n-samples 10000
whest run --estimator estimator.py --dataset ./my-eval
```

See [Use Evaluation Datasets](./use-evaluation-datasets.md) for details.

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
| `--n-samples N` | Ground-truth sampling samples per MLP. The contest default (in `whest run` without an explicit override) is `100 * 100 * 256 = 2,560,000`; `whest dataset bake --n-samples` defaults to `10000`. Drop to `--n-samples 5000` for a ~10x faster local sanity check; raise back up before drawing real conclusions. |
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

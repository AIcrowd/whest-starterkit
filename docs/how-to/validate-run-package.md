# Validate, Run, and Package

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page for the standard local participant loop.

## 🚀 Do this now

Validate estimator loading and output contract:

> `whest validate` is a fast sanity check using a small fixed MLP (width=4, depth=2). It verifies loading, that `setup(context)` finishes inside the graded 5 s cap, output shape, and value finiteness, not full behavioral or performance correctness. Always follow with `whest run` for realistic tests.

```bash
uv run whest validate --estimator estimator.py
```

Run local scoring (recommended default runner):

```bash
uv run whest run --estimator estimator.py
```

`whest run` defaults to `--runner local` for fast iteration.

Run against the published evaluation dataset on HuggingFace (skips sampling; much faster for repeated runs, no local bake needed):

```bash
uv run whest run \
    --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2
# auto-resolves to the `mini` split (100 MLPs, cached after the first call)
```

> ⚠ Check the current revision tag before relying on this command; see
> [Use Evaluation Datasets](./use-evaluation-datasets.md) for the coordinates in
> force. `whest run` without `--dataset` generates MLPs locally and needs none.

Or bake a custom local dataset once and reuse it:

```bash
uv run whest dataset bake --output ./my-eval --n-mlps 10 --n-samples 100_000 --width 1024 --depth 16
# ~80 s and ~670 MB for 10 MLPs at 1024x16 on a laptop; sampling time scales
# linearly in --n-samples
uv run whest run --estimator estimator.py --dataset ./my-eval
```

See [Use Evaluation Datasets](./use-evaluation-datasets.md) for details.

Run the way the grader does, with an isolated subprocess and the memory cap enforced:

```bash
uv run whest run --estimator estimator.py --runner subprocess
```

Under the default `--runner local` the 8 GB limit is advisory only; the CLI says so.

Run with machine-readable output:

```bash
uv run whest run --estimator estimator.py --format json
```

`--json` still works as an alias, but `--format rich|plain|json` is the canonical output selector across the CLI.

Package submission artifact (single file, the common case):

```bash
uv run whest package --estimator estimator.py --output ./submission.tar.gz
```

A **file** argument ships **only that file**; a **folder** argument (`--estimator .`) ships every file in that folder. Either way `whest package` previews exactly what will be submitted and (in folder mode) asks for confirmation before writing; pass `--yes` / `-y` to skip the prompt in CI or scripts. Credential files (`.env`, `*.pem`, keys, …) are never included. The 50 MiB / 50-file caps apply; use `.whestignore` to exclude scratch or large artefacts.

Shipping helper modules or precomputed weights? Keep them in the folder and package the folder; they ship by being present, no extra flags. Data files (weights, lookup tables, precomputed artifacts) are explicitly permitted; **third-party packages are not**. Vendoring numpy, scipy, a BLAS build, or any compiled kernel into your folder is prohibited by the Phase 2 rules and disqualifiable (not merely unavailable), so do that work offline and ship the result as data. See [Allowed Code](../concepts/allowed-code.md) and [Ship Weights and Multi-File Submissions](./ship-weights.md).

## Useful `whest run` flags

These all show up in `whest run --help` but get lost there. Reach for them when:

| Flag | Reach for it when… |
|---|---|
| `--seed N` | Deterministic comparison between two estimator versions. Pin the seed and the same MLPs, the same per-MLP `mlp.seed` values, and the same `SetupContext.seed` are used across runs. Also accepted by `whest validate` (seeds the validation `setup(ctx)` call). With `--dataset`, the dataset supplies the per-MLP seeds and `--seed` controls `ctx.seed` only. |
| `--n-samples N` | Ground-truth Monte-Carlo samples per MLP, used **only** for `--dataset`-less runs; the default is **200,000** (`whestbench.cli._default_contest_spec`). With `--dataset` the flag is ignored: the published Phase 2 dataset bakes `n_samples = 1_000_000_000` per MLP. Drop to `--n-samples 5000` for a ~9x faster local sanity check (measured 12.9 s → 1.4 s for `--n-mlps 1` at 1024×16); raise back up before drawing real conclusions. `whest dataset bake` has no `--n-samples` default: it is a required flag; see [Use Evaluation Datasets](./use-evaluation-datasets.md#-bake-your-own-rare). |
| `--n-mlps N` | Default `10` **without** `--dataset`. **With** `--dataset` it defaults to the whole split (100 for `mini`, 1000 for `full`) and is clamped to the split size, so pass it explicitly when you want a short dataset run. Drop to `3` while iterating to cut runtime to about a third; raise to `20+` when you're trying to reduce noise on a close score. |
| `--flop-budget N` | Deliberately *not* matching the grader. Phase 2 grades each MLP against `B_m = 2,199,023,255,552` FLOPs (2^41), and what the budget caps is `C_m = F_m`: your analytical FLOP count, nothing else. **Default 2199023255552, already the graded budget**, so a plain run mirrors the grader without this flag. Bump it to `1e13` to confirm an algorithm idea isn't budget-starved before you optimize for budget, or drop it to an earlier round's value (`2.72e11` for Phase 1, `6.8e10` for `v1-warmup`; see [Competition Rounds](../reference/rounds.md)). The value you were actually scored under is `run_config.flop_budget` in the run's own report. |
| `--profile` | Emits a per-namespace FLOP/time breakdown so you can see where your estimator burns the budget. |
| `--show-diagnostic-plots` | Renders convergence and per-layer error plots inline (terminal-friendly). Pairs well with `--profile`. |
| `--max-threads N` | Pin the BLAS thread pool size so `wall_time_s` is comparable across machines. Useful when triaging a "fast on my laptop, slow in CI" report. **Reach for `--max-threads 1` as your conservative check:** the rules guarantee no particular evaluation hardware, so anything that finishes inside the 120 s per-MLP cap locally only because it is multi-threaded is a bet, not a plan. (This pins the flopscope backend's threads, which is a timing knob only. Your own estimator may not start threads at all; see the Phase 2 allowed-code rules in the [Pre-Submission Checklist](./pre-submission-checklist.md#allowed-code). On the grader the solution process gets 2 pinned vCPUs and the flopscope backend 14.) See [Is scoring hardware-dependent?](../troubleshooting/faq.md#is-scoring-hardware-dependent). |
| `--detail {raw,full}` | `raw` strips Rich formatting (handy for `tee`-ing logs); `full` adds the per-MLP raw arrays. |
| `--wall-time-limit S` | Cap each `predict()` call's wall time. **Default 120.0, already the graded cap**, so you never pass this to match the grader; pass it only to go *stricter* while chasing a numerical edge case that hangs. Confirm the value you ran under in `run_config.wall_time_limit_s`. |
| `--setup-timeout S` | Testing that `setup()` fits the graded budget. **Default 5.0, the graded cap**, and the only limit whose breach fails the **entire submission** rather than one MLP. `whest validate` times `setup()` against the same 5 s cap, though it has no flag of its own. Lower it (`--setup-timeout 2`) to leave headroom for a slower grader if you load weights in `setup()`. Confirm the value you ran under in `run_config.setup_timeout_s`. See [Ship Weights](./ship-weights.md). |
| `--residual-wall-time-limit S` | Cap time spent outside flopscope ops (Python plumbing, loops, control flow). **Default 0.4, already the graded cap**, so a plain `whest run` already gates you exactly as the live round does; exceed it and that MLP's prediction is replaced with zeros. Residual time is not priced into your score any more, and it is not a place to compute: meaningful work outside flopscope ops is disqualifiable. |
| `--debug` + `--fail-fast` | First exception → halt + raw traceback. Combine for the fastest "what broke?" loop. |

## ✅ Expected outcome

- `validate` passes,
- `run` produces a score report,
- `package` creates a `.tar.gz` artifact.

## ⚠️ Common first failure

Symptom: `run` fails after `validate` passed.

A plain `whest run` keeps going after a per-MLP failure and collects the messages into one table at the end:

```text
╭────────────────────────────── Estimator Errors ──────────────────────────────╮
│ 2 of 2 MLP(s) raised during predict.                                         │
│ ┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │
│ ┃ MLP                ┃ Code          ┃ Message                             ┃ │
│ ┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩ │
│ │ megan-chang (#0)   │ PREDICT_ERROR │ index 17 is out of bounds for axis  │ │
│ │                    │               │ 0 with size 16                      │ │
│ └────────────────────┴───────────────┴─────────────────────────────────────┘ │
│ Rerun with --debug to include full tracebacks; --fail-fast to stop on first  │
│ error.                                                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Use this escalation flow:

1. Stop at the first failure and get the raw traceback:

```bash
uv run whest run --estimator estimator.py --fail-fast --debug
```

2. If it only breaks under isolation, reproduce it the way the grader runs it:

```bash
uv run whest run --estimator estimator.py --runner subprocess --fail-fast --debug
```

For runner modes, see [Stage 3: Run Locally](../getting-started/stage-3-run-local.md), [Stage 4: Subprocess Runner](../getting-started/stage-4-run-subprocess.md), and the [Debugging Checklist](./debugging-checklist.md).

## ➡️ Next step

- [Use Evaluation Datasets](./use-evaluation-datasets.md)
- [CLI Reference](../reference/cli-reference.md)
- [Score Report Fields](../reference/score-report-fields.md)

# CLI Reference

> [← Documentation](../README.md)

The `whest` CLI is shipped by [whestbench](https://github.com/AIcrowd/whestbench). The authoritative reference lives there:

→ [whestbench: docs/reference/cli-reference.md](https://github.com/AIcrowd/whestbench/blob/main/docs/reference/cli-reference.md)

## Quick lookup

| Command | What it does | Stage |
|---|---|---|
| `whest smoke-test` | Built-in estimator end-to-end; proves the install works | before 1 |
| `whest init [path]` | Write the starter `estimator.py` + `.whestignore` | 1 |
| `whest validate` | Check estimator contract | 2 |
| `whest dataset bake/download/info/merge` | Build or fetch a real evaluation dataset | 3-4 |
| `whest run --runner local` | Score in-process | 3 |
| `whest run --runner subprocess` | Score in subprocess | 4 |
| `whest package` | Build submission archive. A **file** ships just that file, renamed to `estimator.py` inside the archive; a single file that imports a sibling module is refused at package time (package the folder instead). A **folder** ships the whole folder and must contain `estimator.py`. | 5 |
| `whest validate-package <archive>` | Check a built `.tar.gz` against its manifest before you spend a submission slot | 5 |
| `whest login` | Store your AIcrowd API key; do this before `whest submit` | 5 |
| `whest submit` | Package (if `--estimator` given) and upload to AIcrowd | 5 |
| `whest doctor` | Diagnose environment issues | any |
| `whest profile-simulation` | Benchmark backend correctness/timing | any |
| `whest version` | Print the installed whestbench version | any |

> **One Phase 2 thing the CLI won't tell you.** `whest submit` is capped at
> **10 submissions per team per UTC day**.

## `whest run` limit flags

Every graded limit is already the whestbench 0.16.0 default. The ground truth is
the one thing a local run does not match:

| Graded limit | `whest run` flag | 0.16.0 default | Matches grader? |
|---|---|---|---|
| FLOP budget per MLP | `--flop-budget` | `2199023255552` (`2**41`) | yes |
| Wall clock per `predict()` | `--wall-time-limit` | `120.0` | yes |
| Residual wall time per MLP | `--residual-wall-time-limit` | `0.4` | yes |
| `setup()` wall clock, per run | `--setup-timeout` | `5.0` | yes |
| Price of residual time (λ) | `--lambda-flops-per-second` | `0.0` — residual time is gated, not priced | yes |
| Ground-truth draws per MLP | `--n-samples` (dataset-less) | `200_000` | no — the Phase 2 dataset was baked at 1e9 |

So a bare `uv run whest run --estimator estimator.py --runner local` already
rehearses every graded limit; the flags exist to *change* them.
`--no-residual-wall-time-limit` disables the residual gate entirely. You need it
only to re-score an earlier round, and never on its own: that round's budget,
lambda, wall cap and dataset revision all have to be restored together, or the
run is scored under a mix of two rulebooks and the number matches neither. The
complete per-round recipe is in [Competition Rounds](rounds.md#reproducing-a-score-from-an-earlier-round).
Confirm what you actually ran under with `--format json` and read `run_config`.

See [Competition Rounds](rounds.md) for every round's values side by side, and
[Estimator Contract: Phase 2 limits](estimator-contract.md#phase-2-limits) for
what each cap does when you cross it.

## `whest run` dataset flags

| Flag | What it does |
|---|---|
| `--dataset` | A baked dataset directory, or `hf://owner/repo[@revision]` |
| `--revision` | HF Hub revision (tag or commit SHA) for `--dataset` — the long form of the `@revision` suffix |
| `--split` | Which split to evaluate; required only when the dataset has more than one |
| `--streaming` | Iterate the dataset from HF instead of downloading it. Iteration-only (no random access) and **nothing is cached**, so every run re-fetches — use it for small `--n-mlps` debugging runs, not for scoring. |

Pin the revision on every dataset run. See
[Use Evaluation Datasets](../how-to/use-evaluation-datasets.md).

# CLI Reference

> [← Documentation](../README.md)

The `whest` CLI is shipped by [whestbench](https://github.com/AIcrowd/whestbench). The authoritative reference lives there:

→ [whestbench: docs/reference/cli-reference.md](https://github.com/AIcrowd/whestbench/blob/main/docs/reference/cli-reference.md)

## Quick lookup

| Command | What it does | Stage |
|---|---|---|
| `whest validate` | Check estimator contract | 2 |
| `whest run --runner local` | Score in-process | 3 |
| `whest run --runner subprocess` | Score in subprocess | 4 |
| `whest package` | Build submission archive — a **file** ships just that file, a **folder** ships the whole folder | 5 |
| `whest submit` | Package (if `--estimator` given) and upload to AIcrowd | 5 |
| `whest doctor` | Diagnose environment issues | any |
| `whest profile-simulation` | Benchmark backend correctness/timing | any |

> **Two Phase 2 things the CLI won't tell you.** `whest submit` is capped at
> **10 submissions per team per UTC day**. And `whest run`'s local defaults
> are not the grader's — pass
> `--flop-budget 2199023255552 --wall-time-limit 120 --residual-wall-time-limit 0.4`
> when you want a faithful rehearsal. See
> [Estimator Contract: Phase 2 limits](estimator-contract.md#phase-2-limits).

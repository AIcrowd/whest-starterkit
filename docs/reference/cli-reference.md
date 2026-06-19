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

# CLI Reference

The `whest` CLI is shipped by [whestbench](https://github.com/AIcrowd/whestbench). The authoritative reference lives there:

→ [whestbench: docs/reference/cli-reference.md](https://github.com/AIcrowd/whestbench/blob/main/docs/reference/cli-reference.md)

This kit pins a specific version of whestbench. Check `pyproject.toml` for the tag in use.

## Quick lookup

| Command | What it does | Stage |
|---|---|---|
| `whest validate` | Check estimator contract | 2 |
| `whest run --runner local` | Score in-process | 3 |
| `whest run --runner subprocess` | Score in subprocess | 4 |
| `whest run --runner docker` | Score in docker (coming soon) | 5 |
| `whest package` | Build submission tarball | 6 |
| `whest doctor` | Diagnose environment issues | any |

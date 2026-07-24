# `whest doctor`

> [← Documentation](../README.md)

## 🎯 When to use this page

You ran `uv run whest doctor` and want to interpret a `[WARN]` or
`[FAIL]` row, or you're chasing a "works on my laptop, fails in CI" bug
and want to know what `doctor` actually checks.

## 🚀 Run it

```bash
uv run whest doctor
```

```
Running whestbench doctor
whest doctor
------------
  [OK]    Python version         3.10.20 satisfies >=3.10
  [OK]    uv on PATH             /Users/mohanty/.local/bin/uv
  [OK]    whest install mode     tool-installed · 0.13.0
  [WARN]  BLAS thread pool       no BLAS pool detected
                                 threadpoolctl detected no BLAS pool; numpy may be using a fallback. Usually harmless.
  [OK]    Free disk in CWD       73.3 GiB free
  [OK]    CWD writable           /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/whest-starterkit

  6 checks · 5 ok · 1 warn · 0 fail
✓ Ran 6 checks in 10ms
```

Add `--strict` to make warnings fail the exit code (useful in CI).
Add `--format json` for a machine-readable record.

## Checks

| Check | What it verifies | Fix if `[WARN]` / `[FAIL]` |
|---|---|---|
| **Python version** | Interpreter is Python 3.10+ as required by `pyproject.toml`. | Install Python 3.10+ (`uv python install 3.10`) and rerun `uv sync`. |
| **uv on PATH** | `uv` binary is reachable. Required for the `uv run …` commands the docs use. | Install uv: <https://docs.astral.sh/uv/getting-started/installation/>. |
| **whest install mode** | Reports how `whestbench` was installed: `tool-installed · <version>` (published wheel/pip install), `from-source · <path>` (non-editable source checkout), or `editable · <path>` (editable/dev install). | None — informational. `tool-installed` is expected for the standard starter-kit dependency stack. |
| **BLAS thread pool** | `threadpoolctl` finds a BLAS pool (OpenBLAS, MKL, Accelerate). When absent, numpy falls back to a less-optimized path; FLOPs counts are unaffected, but wall-clock time may be higher and `--max-threads` has no effect. | Install a BLAS-backed numpy (the default `pip install numpy` usually has one). On macOS Apple Silicon, `pip install numpy` ships Accelerate. The warning is harmless for scoring; only matters if you're optimizing wall time. |
| **Free disk in CWD** | At least a few GiB free. Datasets, logs, and the per-run report can grow on long iteration sessions. | Clear space; `whest run` may fail to write reports otherwise. |
| **CWD writable** | The directory you're invoking `whest` from is writable. | `cd` into a writable directory or fix permissions. |

## Reading the summary line

```
  6 checks · 5 ok · 1 warn · 0 fail
```

- `fail` → exit code is non-zero and you have a real problem.
- `warn` → exit code is `0` by default; pass `--strict` to flip warnings into failures (useful for CI, not for daily iteration).
- `ok` → fully green.

## ➡️ See also

- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
- [CLI Reference](./cli-reference.md)

# `whest doctor`

> [← Documentation](../README.md)

## 🎯 When to use this page

You ran `uv run whest doctor` and want to interpret a `[WARN]` or
`[FAIL]` row, or you are debugging a difference between your machine and CI
and want to know what `doctor` checks.

## 🚀 Run it

```bash
uv run whest doctor
```

```
Running whestbench doctor
uv run whest doctor
------------
  [OK]    Python version         3.10.20 satisfies >=3.10
  [OK]    uv on PATH             /home/you/.local/bin/uv
  [OK]    whest install mode     tool-installed · 0.16.0
  [WARN]  BLAS thread pool       no BLAS pool detected
                                 threadpoolctl detected no BLAS pool; numpy may be using a fallback. Usually harmless.
  [OK]    Free disk in CWD       73.3 GiB free
  [OK]    CWD writable           /home/you/whest-starterkit

  6 checks · 5 ok · 1 warn · 0 fail
✓ Ran 6 checks in 10ms
```

To make warnings fail the exit code, add `--strict`. For a machine-readable
record, add `--format json`.

## Checks

| Check | What it verifies | Fix if `[WARN]` / `[FAIL]` |
|---|---|---|
| **Python version** | Interpreter satisfies **whestbench's own** `Requires-Python` (currently `>=3.10`), read from the installed distribution metadata, not from this kit's `pyproject.toml`. The detail line echoes the exact specifier it checked, and appends `(fallback floor)` if whestbench's metadata could not be read. | Install Python 3.10+ (`uv python install 3.10`) and rerun `uv sync`. |
| **uv on PATH** | `uv` binary is reachable. Required for the `uv run …` commands the docs use. | Install uv: <https://docs.astral.sh/uv/getting-started/installation/>. |
| **whest install mode** | Reports how `whestbench` was installed: `tool-installed · <version>` (published wheel/pip install), `from-source · <path>` (non-editable source checkout), or `editable · <path>` (editable/dev install). | None — informational. `tool-installed` is expected for the standard starter-kit dependency stack. |
| **BLAS thread pool** | `threadpoolctl` can enumerate a BLAS pool (OpenBLAS, MKL). A `[WARN]` means detection failed, which is not the same as numpy lacking BLAS: Accelerate on Apple Silicon is invisible to threadpoolctl. | **On macOS Apple Silicon this WARN is expected and needs no action.** numpy there links against Accelerate, which `threadpoolctl` cannot enumerate, so `threadpool_info()` returns `[]` and the check warns even though BLAS is fully active. Confirm with `uv run python -c "import numpy as np; np.show_config()"` and look for `blas: name: accelerate`. On Linux an empty pool list is real: reinstall numpy from a wheel (`uv pip install --force-reinstall numpy`). Either way your FLOP counts are analytical and unaffected. Wall time still matters: each MLP has a 120 s wall cap and a 400 ms residual cap, and exceeding either zeroes that MLP's predictions. `--max-threads` has no effect while no pool is detected. See [Is scoring hardware-dependent?](../troubleshooting/faq.md#is-scoring-hardware-dependent). |
| **Free disk in CWD** | At least **1.0 GiB** free (`_MIN_FREE_GIB` in whestbench's `doctor.py`). Below that you get a `[WARN]`, never a `[FAIL]`; the only way this row fails is if the disk-usage read itself errors. Datasets, logs, and per-run reports can still grow well past 1 GiB on long iteration sessions. | Clear space; `whest run` may fail to write reports otherwise. |
| **CWD writable** | The directory you're invoking `whest` from is writable. | `cd` into a writable directory or fix permissions. |

## Reading the summary line

```
  6 checks · 5 ok · 1 warn · 0 fail
```

- `fail` → exit code is non-zero; the check found a problem you need to fix.
- `warn` → exit code is `0` by default; to turn warnings into failures, pass `--strict` (suited to CI, not to daily iteration).
- `ok` → the check passed.

## ➡️ See also

- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
- [CLI Reference](./cli-reference.md)

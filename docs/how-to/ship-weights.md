# Ship Weights and Multi-File Submissions

> [← Documentation](../README.md)

## When to use this page

Use this page when you want to pre-compute something offline (e.g. a calibration
scalar, a learned projection matrix, a lookup table) and load it inside
`setup()` — or when your estimator spans more than one Python module.

---

## (a) Splitting code across modules

To ship more than one file, package the **folder** — point `--estimator` at the
directory, not at `estimator.py`:

```bash
uv run whest package --estimator . --output submission.tar.gz
```

Folder mode bundles every non-ignored file in the folder, so helper modules and
data files next to `estimator.py` ship and import on the grader with the same
paths as locally:

```
my-submission/
  estimator.py       ← entry point
  helper.py          ← imported by estimator.py → ships (folder mode)
  layers.py          ← same
  weights.npz        ← data file → ships (folder mode)
```

`whest package` lists every file it will ship and asks you to confirm before
writing, and warns if a `.py` file isn't reachable by import from `estimator.py`
(likely a scratch file you forgot to exclude).

> ⚠ Packaging the file alone — `whest package --estimator estimator.py` — ships
> **only `estimator.py`**; `helper.py` and `weights.npz` would be left out. For a
> multi-file submission, always point `--estimator` at the folder.

---

## (b) Authoring weights offline with `flopscope.Module`

flopscope only loads **pickle-free** array weights: `fnp.load` and
`flopscope.Module` both use `np.load(allow_pickle=False)`, so a pickled model
(`torch.save`, `joblib`, `pickle`) will **not** load on the grader. Author your
weights as a `flopscope.Module` — public array attributes are saved and restored
automatically:

```python
import flopscope
import flopscope.numpy as fnp

class Weights(flopscope.Module):
    def __init__(self) -> None:
        self.scale = fnp.ones(())     # public array attribute → saved & restored

# Offline compute is free — only predict()-time FLOPs count toward your score.
w = Weights()
w.scale = fnp.asarray(2.0)            # replace with your real precomputation
w.save("weights.npz")                 # plain .npz, no pickle
```

`.save()` writes a plain `.npz`; nested `Module`s and lists/tuples/dicts of arrays
are flattened automatically. (For a single bare array you can also `np.savez` /
`fnp.load` directly, but `Module` keeps multi-array weights structured and reloads
them into a typed object.)

**Designing the weights themselves** — which operations are free vs FLOP-counted,
and the full `fnp` module surface (array creation, RNG, reductions, matmul,
einsum) — is covered in the [Flopscope Primer](../reference/flopscope-primer.md)
and [Code Patterns](../reference/code-patterns.md).

---

## (c) Loading in `setup()` via `submission_dir`

`context.submission_dir` is set by the runner to the folder containing your
`estimator.py` — both locally (`whest validate` / `whest run`) and on the
grader (the extracted submission root).  Always guard against `None` before
constructing a `Path` from it — it is `None` outside the runner context:

```python
from pathlib import Path
import flopscope
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext

class Weights(flopscope.Module):
    def __init__(self) -> None:
        self.scale = fnp.ones(())

class Estimator(BaseEstimator):
    def setup(self, context: SetupContext) -> None:
        self._weights = None
        if context.submission_dir is not None:
            weights_path = Path(context.submission_dir) / "weights.npz"
            if weights_path.exists():
                self._weights = Weights.from_file(str(weights_path))  # 0 FLOPs
```

`from_file` (and `fnp.load`) cost **0 FLOPs** — loading data does not count
against your budget. **Pass a `str` path, not a `Path`** — the grader's
`flopscope-client` requires a string filename. (The full flopscope in your local
venv also accepts a `Path`, so a `Path` appears to work under `whest validate` but
fails on the grader — always wrap with `str(...)`.)

See the full worked example at [`examples/04_shipped_weights.py`](../../examples/04_shipped_weights.py).

---

## (d) Caps and `.whestignore`

`whest package` enforces two hard caps:

| Cap | Limit |
|-----|-------|
| Total submission size | 50 MiB (the CLI error reports this as ~52 MB) |
| Total file count | 50 files |

If your folder contains large scratch files, cached datasets, or other
artefacts you don't want to ship, list them in `.whestignore` next to
`estimator.py` (same glob syntax as `.gitignore`):

```
# .whestignore
*.egg-info/
scratch/
debug_weights.pkl
```

`whest init` creates a starter `.whestignore` for you. The built-in ignore list
already excludes common non-submission artefacts (`.git/`, `__pycache__/`,
`*.pyc`, etc.) and **all credential files** (`.env`, `*.pem`, `*.key`, private
keys) for security, so you only need to add project-specific entries.

---

## (e) Package preview, `--yes`, and dry run

Folder mode gives you **full visibility** before anything ships: `whest package`
lists every file and asks you to confirm:

```
Packaging folder ./ → submission-20260610-120000.tar.gz
Everything in this folder will be submitted, except .gitignore / .whestignore
matches and credential files (excluded for security).
Submitting 3 files (42.3 KB):
  estimator.py  (1.2 KB)
  helper.py     (0.9 KB)
  weights.npz   (40.2 KB)
Submit all 3 files (42.3 KB)? [y/N]
```

Skip the prompt in CI with `--yes` / `-y`:

```bash
whest package --estimator . --yes
```

To preview **without** building the archive or uploading anything:

```bash
whest submit --estimator . --dry-run
```

This shows the full file list, sizes, and versions, then stops.

---

## (f) Grader timing note

The grader measures **wall time over your entire submission process** — imports,
`setup()`, and every `predict()` call. Keep `setup()` to cheap operations:
load files, unpack arrays, set up data structures. Do not train a model in
`setup()`.

The right pattern is to do all heavy computation **offline** (before you
package), save the result to a file, and load it in `setup()`. That load is
fast, pickle-free, and costs 0 FLOPs.

---

## ➡️ Next step

- [Pre-Submission Checklist](./pre-submission-checklist.md)
- [Validate, Run, and Package](./validate-run-package.md)
- [Stage 5: Package Your Submission](../getting-started/stage-5-package.md)

# Ship Weights and Multi-File Submissions

> [← Documentation](../README.md)

## When to use this page

Use this page when you want to pre-compute something offline (e.g. a calibration
scalar, a learned projection matrix, a lookup table) and load it inside
`setup()` — or when your estimator spans more than one Python module.

---

## (a) Splitting code across modules

`whest package` bundles your estimator's **entire folder**. If you keep helper
modules next to `estimator.py`, they ship automatically and are importable on
the grader with the same import paths as locally:

```
my-submission/
  estimator.py       ← entry point
  helper.py          ← imported by estimator.py → ships automatically
  layers.py          ← same
  weights.npz        ← data file → ships automatically
```

No extra flags needed — the packaging step collects every non-ignored file in
the folder. `whest package` also warns you if a `.py` file in the folder is
**not** reachable by import from `estimator.py` (likely a scratch file you
forgot to exclude).

---

## (b) Authoring `weights.npz` offline

Offline compute is free — only `predict()`-time FLOPs count toward your score.
Pre-compute anything expensive outside the challenge runner:

```python
import numpy as np

scale = np.float32(2.0)           # replace with your real computation
np.savez("weights.npz", scale=scale)
```

Plain NumPy `.npz` is the recommended format: no special dependencies, no
pickle, loads without registered weights classes.

---

## (c) Loading in `setup()` via `submission_dir`

`context.submission_dir` is set by the runner to the folder containing your
`estimator.py` — both locally (`whest validate` / `whest run`) and on the
grader (the extracted submission root).  Always guard against `None` before
constructing a `Path` from it — it is `None` outside the runner context:

```python
from pathlib import Path
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext

class Estimator(BaseEstimator):
    def setup(self, context: SetupContext) -> None:
        scale = None
        if context.submission_dir is not None:
            weights_path = Path(context.submission_dir) / "weights.npz"
            if weights_path.exists():
                scale = fnp.load(str(weights_path))["scale"]  # 0 FLOPs to load
        self._scale = scale if scale is not None else fnp.ones(())
```

`fnp.load` (flopscope's NumPy-compatible load) costs **0 FLOPs** — loading
data does not count against your budget. **Pass a `str` path, not a `Path`** —
the grader's `flopscope-client` requires a string filename. (The full flopscope
in your local venv also accepts a `Path`, so a `Path` appears to work under
`whest validate` but fails on the grader — always wrap with `str(...)`.)

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
`*.pyc`, etc.), so you only need to add project-specific entries.

---

## (e) Package preview, `--yes`, and dry run

Before packaging, `whest package` shows a file/size preview and asks for
confirmation:

```
● Packaging estimator.py → submission-20260610-120000.tar.gz
  Files to bundle (3 files, 42.3 KB):
    estimator.py  (1.2 KB)
    helper.py     (0.9 KB)
    weights.npz   (40.2 KB)
  Package these 3 files (42.3 KB)? [y/N]
```

Skip the prompt in CI with `--yes` / `-y`:

```bash
whest package --estimator estimator.py --yes
```

To preview **without** building the archive or uploading anything:

```bash
whest submit --estimator estimator.py --dry-run
```

This shows the full manifest (files, sizes, versions) and then stops.

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

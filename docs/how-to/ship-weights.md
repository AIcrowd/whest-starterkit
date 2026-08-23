# Ship weights and multi-file submissions

> [← Documentation](../README.md)

## When to use this page

Use this page when you want to pre-compute something offline (for example, a
calibration scalar, a learned projection matrix, or a lookup table) and load it inside
`setup()`, or when your estimator spans more than one Python module.

> **Where the Phase 2 rules put this.** Shipping **data** (weights, lookup
> tables, precomputed artifacts) is explicitly permitted, and it costs less
> than it did in Phase 1: with `C_m = F_m`, everything you compute before
> you package adds nothing to your score. Shipping **code** is what the rules
> restrict. Vendored numpy/scipy/BLAS, compiled kernels of any kind, and anything
> reached through `ctypes`/`cffi` are prohibited outright, and a submission
> carrying them is disqualifiable rather than merely broken; see
> [Allowed Code](../concepts/allowed-code.md) for the full rule.
>
> The Sponsor also reserves the discretion to decline to treat a bundled file
> as data. A file that is really a program (bytecode, a serialized kernel, an
> instruction table your estimator dispatches through) can be ruled out on
> those grounds whatever extension it arrives with. Arrays of numbers your
> estimator does arithmetic on are what this page is about; if your artifact
> is something else, ask [arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com) before you build a
> submission around it.

---

## (a) Splitting code across modules

To ship more than one file, package the **folder**. Point `--estimator` at the
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

> ⚠ Packaging the file alone (`whest package --estimator estimator.py`) ships
> **only that file**, renamed to `estimator.py` inside the archive whatever you
> called it locally. Since whestbench 0.16.0, if `estimator.py` imports a
> sibling module the command **refuses** rather than producing an archive that
> would `ImportError` on the grader:
>
> ```
> ⚠ Single-file submission: only estimator.py will be submitted. 2 items beside it will NOT be included: helper.py, weights.npz.
> Error [package:PACKAGING_VALIDATION_ERROR]: estimator.py imports helper, which lives beside it and will NOT be bundled ...
> ```
>
> Data files like `weights.npz` are only *named* in that warning, not protected
> by it. They are dropped. For anything multi-file, point `--estimator` at the
> folder.

---

## (b) Authoring weights offline with `flopscope.Module`

flopscope only loads **pickle-free** array data. Locally `fnp.load` (and
`flopscope.Module`, which goes through it) calls `np.load(allow_pickle=False)`;
on the grader flopscope-client parses the `.npy`/`.npz` with its own codec and
cannot unpickle at all. Either way a pickled model (`torch.save`, `joblib`,
`pickle`) will **not** load. Author your weights as a `flopscope.Module`;
flopscope saves and restores public array attributes automatically:

```python
import flopscope
import flopscope.numpy as fnp

class Weights(flopscope.Module):
    def __init__(self) -> None:
        self.scale = fnp.zeros((), dtype=fnp.float32)  # public array attribute → saved & restored

# Offline compute is free — only predict()-time FLOPs count toward your score,
# and in Phase 2 those FLOPs are the whole of it: C_m = F_m.
w = Weights()
w.scale = fnp.asarray(2.0)            # replace with your real precomputation
w.save("weights.npz")                 # plain .npz, no pickle
```

`.save()` writes a plain `.npz` and flattens nested `Module`s and lists, tuples,
and dicts of arrays automatically. (For a single bare array you can also `np.savez` /
`fnp.load` directly, but `Module` keeps multi-array weights structured and reloads
them into a typed object.) Make the save call offline with plain `numpy` (`np.savez`): inside `predict()`, `fnp.save`/`fnp.savez` bill an egress cost since flopscope 0.9 (≈4 FLOPs per element × dtype rate, plus small headers), while `fnp.load` stays free.

The [Flopscope Primer](../reference/flopscope-primer.md) and
[Code Patterns](../reference/code-patterns.md) cover **designing the weights
themselves**: which operations are free vs FLOP-counted, and the full `fnp`
module surface (array creation, RNG, reductions, matmul, einsum).

---

## (c) Loading in `setup()` via `submission_dir`

The runner sets `context.submission_dir` to the folder containing your
`estimator.py`, both locally (`whest validate` / `whest run`) and on the
grader (the extracted submission root).  Always guard against `None` before
constructing a `Path` from it; it is `None` outside the runner context:

```python
from pathlib import Path
import flopscope
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext

class Weights(flopscope.Module):
    def __init__(self) -> None:
        self.scale = fnp.zeros((), dtype=fnp.float32)

class Estimator(BaseEstimator):
    def setup(self, context: SetupContext) -> None:
        self._weights = None
        if context.submission_dir is not None:
            weights_path = Path(context.submission_dir) / "weights.npz"
            if weights_path.exists():
                self._weights = Weights.from_file(str(weights_path))  # 0 FLOPs
```

`fnp.load` costs **0 FLOPs**, and `from_file` adds only whatever your `__init__`
bills; it re-runs the constructor before restoring state, so keep that
constructor to `fnp.zeros`. Reading the file itself never counts against your
budget. **Pass a `str` path, not a `Path`**: the grader's
`flopscope-client` requires a string filename. (The full flopscope in your local
venv also accepts a `Path`, so a `Path` appears to work under `whest validate` but
fails on the grader; always wrap with `str(...)`.)

See the full worked example at [`examples/04_shipped_weights.py`](../../examples/04_shipped_weights.py).

---

## (d) Caps and `.whestignore`

`whest package` enforces two hard caps:

| Cap | Limit |
|-----|-------|
| Total submission size | 50 MiB (the CLI error reports this as ~52 MB) |
| Total file count | 50 files |

Width 1024 makes that first cap tighter than it looks: one `(1024, 1024)`
float32 matrix is 4 MiB, so a dozen of them is the whole allowance. Ship the
smallest sufficient artifact (a projection basis, a rank-*k* factor, a
calibration table) rather than a full-rank matrix per layer.

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
already excludes common non-submission artefacts such as `.git/`,
`__pycache__/`, and `*.pyc`, plus **all credential files** (`.env`, `*.pem`,
`*.key`, private keys) for security, so you only need to add project-specific
entries.

---

## (e) Package preview, `--yes`, and dry run

Folder mode gives you **full visibility** before anything ships: `whest package`
lists every file and asks you to confirm:

```
Packaging folder /home/you/my-submission/ → submission-20260610-120000.tar.gz
Everything in this folder will be submitted, except .gitignore / .whestignore
matches and credential files (excluded for security).
Submitting 3 files (42.3 KB):
  estimator.py  (1.2 KB)
  helper.py  (0.9 KB)
  weights.npz  (40.2 KB)
Submit all 3 files (42.3 KB)? [y/N]
```

Skip the prompt in CI with `--yes` / `-y`:

```bash
uv run whest package --estimator . --yes
```

To preview and self-check without writing an archive into your folder or
uploading anything:

```bash
uv run whest submit --estimator . --dry-run
```

It packages into a temporary directory, runs the same `validate-package`
integrity check `whest submit` runs before upload, prints the file list, sizes,
the resolved `flopscope==` / `whestbench==` versions, and the archive size it
would have uploaded, then discards it.

---

## (f) Grader timing note

The grader applies two separate time limits. Your module import plus
`setup()` share a hard **5 s** budget, spent **once per worker process, not
once per submission**. Several workers serve one submission (roughly 5-15
setup runs in practice), and since whestbench 0.16.0 the grader replaces a
worker that dies mid-suite, which re-runs your `setup()` and re-spends the 5 s.
Each `predict()` call then gets its own hard **120 s**. The setup limit and the
`predict()` limit are independent of each other.

Setup's cost is not billed: not to the FLOP budget, not to
`residual_wall_time_s`. The limit is short and strict. Five seconds
covers a file read and an unpack, and nothing more; overrunning them fails the
**whole submission** with `SETUP_TIMEOUT`, not one MLP, and takes one of that
day's ten submission slots with it. Time your `setup()` on a cold cache before
you ship, not after. And because setup can run many times, an expensive setup
costs that time on every worker. That is a second reason to load rather than
compute.

So keep `setup()` to cheap operations: load files, unpack arrays, set up data
structures. Do not train a model in `setup()`.

Do all heavy computation **offline** (before you package), save the result to a
file, and load it in `setup()`. That load is fast, pickle-free, and costs
0 FLOPs.

---

## ➡️ Next step

- [Pre-Submission Checklist](./pre-submission-checklist.md)
- [Validate, Run, and Package](./validate-run-package.md)
- [Stage 5: Package Your Submission](../getting-started/stage-5-package.md)

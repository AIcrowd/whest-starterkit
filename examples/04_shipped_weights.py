"""Ship precomputed weights the safe way, with ``flopscope.Module``.

flopscope only loads **pickle-free** array weights (`np.load(allow_pickle=False)`),
so you cannot ship a pickled model. ``flopscope.Module`` is the structured way
to author the ones it does support. Subclass it, set your weights as public array
attributes; then ``.save(path)`` writes a plain ``.npz`` and ``.from_file(path)``
reconstructs the object on the grader, with no pickle and 0 FLOPs to load.

Workflow:
  1. Compute the weights offline (free, off the FLOP budget) and ``.save()`` them.
  2. Copy this file to ``estimator.py`` in a folder of its own (folder submissions
     require that exact filename), keep the ``.npz`` beside it, and package the
     folder: ``whest package --estimator .`` (ships ``estimator.py``,
     ``weights.npz``, ``manifest.json``).
  3. Load with ``Weights.from_file(...)`` in ``setup()``. ``context.submission_dir``
     points at your estimator's folder both locally and on the grader.

See docs/how-to/ship-weights.md for the full walkthrough.
"""

from __future__ import annotations

from pathlib import Path

import flopscope
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

WEIGHTS_FILE = "weights.npz"


class Weights(flopscope.Module):
    """A pickle-free weight bundle. Public (non-underscore) array attributes are
    saved and restored automatically; private (`_`-prefixed) attributes are not."""

    def __init__(self) -> None:
        # Public (non-underscore) array attribute -> saved & restored.
        # `fnp.zeros`, not `fnp.ones`: zeros is on flopscope's free list, while
        # `ones` bills 1 FLOP per element, and at the float64 default that is 2
        # FLOPs even for this 0-d placeholder. Spelling the dtype out keeps the
        # float32 rate. Measured under flopscope 0.12.0: `fnp.ones(())` = 2 FLOPs,
        # `fnp.zeros((), dtype=fnp.float32)` = 0.
        self.scale = fnp.zeros((), dtype=fnp.float32)


def build_weights() -> Weights:
    """Compute the weights offline. This runs outside the challenge runner, so it
    is free; only `predict()`-time FLOPs count toward your score."""
    w = Weights()
    w.scale = fnp.asarray(2.0)  # replace with your real precomputation
    return w


class Estimator(BaseEstimator):
    def setup(self, context: SetupContext) -> None:
        self._weights: Weights | None = None
        if context.submission_dir is not None:
            path = Path(context.submission_dir) / WEIGHTS_FILE
            if path.exists():
                # Pass a str, not a Path: the grader's flopscope-client requires a
                # string filename (the local full-flopscope build also accepts a
                # Path, so a Path "works" locally but fails on the grader).
                # `from_file` is pickle-free and costs 0 FLOPs.
                self._weights = Weights.from_file(str(path))

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        out = fnp.zeros((mlp.depth, mlp.width))
        if self._weights is not None:
            out = out * self._weights.scale
        return out


if __name__ == "__main__":
    import sys
    import tempfile

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    # Author the weights offline and save them into a submission folder, then load
    # them exactly as the grader would. In a real submission the `.npz` lives next
    # to estimator.py and ships when you `whest package --estimator .`.
    submission_dir = tempfile.mkdtemp()
    build_weights().save(str(Path(submission_dir) / WEIGHTS_FILE))

    mlp = build_mlp(width=1024, depth=16, seed=0)  # competition shape
    # The local engine calls setup() for you, exactly as the framework does before
    # predict(). It defaults `submission_dir` to the folder this file lives in; we
    # override it here because the weights above went to a tempdir rather than
    # being committed to the repo. In a real submission you would not pass it;
    # the default already points at the folder holding your estimator and .npz.
    compare_against_monte_carlo(Estimator(), mlp, submission_dir=submission_dir)

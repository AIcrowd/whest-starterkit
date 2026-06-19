"""Ship precomputed weights the safe way, with ``flopscope.Module``.

flopscope only loads **pickle-free** array weights (`np.load(allow_pickle=False)`),
so you cannot ship a pickled model — and ``flopscope.Module`` is the structured way
to author the ones it does support. Subclass it, set your weights as public array
attributes; then ``.save(path)`` writes a plain ``.npz`` and ``.from_file(path)``
reconstructs the object on the grader — no pickle, 0 FLOPs to load.

Workflow:
  1. Compute the weights offline (free — off the FLOP budget) and ``.save()`` them.
  2. Keep the ``.npz`` next to your estimator and package the **folder**
     (`whest package --estimator .`) so it ships.
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
        self.scale = fnp.ones(())  # public array attribute -> saved & restored


def build_weights() -> Weights:
    """Compute the weights offline. This runs outside the challenge runner, so it
    is free — only `predict()`-time FLOPs count toward your score."""
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

    mlp = build_mlp(width=256, depth=32, seed=0)  # phase-1 shape (warmup used depth=8)
    estimator = Estimator()
    # The framework always calls setup() before predict(); do the same here.
    estimator.setup(
        SetupContext(
            width=256,
            depth=32,
            flop_budget=272_000_000_000,
            api_version="1.0",
            submission_dir=submission_dir,
        )
    )
    compare_against_monte_carlo(estimator, mlp)

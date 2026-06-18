"""Random-baseline estimator.

Demonstrates the canonical contract surface (``setup`` / ``predict`` /
``teardown``) *and* the whestbench RNG-seeding contract:

* ``self._setup_rng = fnp.random.default_rng(ctx.seed)`` inside ``setup`` --
  one-time setup RNG seeded from the grader-supplied ``ctx.seed``. Used for
  random precompute that should be deterministic across MLPs and across
  regrades (here: nothing -- this baseline has no setup-time precompute,
  but the scaffold is present so the propagation examples (01–03) all
  demonstrate the pattern).
* ``rng = fnp.random.default_rng(mlp.seed)`` inside ``predict`` -- per-MLP
  RNG seeded from the grader-supplied ``mlp.seed``. This is the seed
  whose determinism the grader checks under regrade. Submissions that
  use their own per-MLP seeds (or unseeded randomness) may be
  disqualified -- see
  ``docs/reference/estimator-contract.md``
  ("Reproducibility under the grader seed") for the contract.
"""

from __future__ import annotations

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._context = None
        self._setup_rng = None  # set from ctx.seed inside setup()

    def setup(self, ctx: SetupContext) -> None:
        self._context = ctx
        # Submission-level RNG: seeded once per run from the grader-supplied
        # ctx.seed. Use for any setup-time random precompute.
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        # Per-MLP RNG seeded from the grader-supplied seed. Identical across
        # regrades, distinct per MLP. ALWAYS seed predict-time randomness
        # from mlp.seed.
        rng = fnp.random.default_rng(mlp.seed)
        return fnp.asarray(
            rng.uniform(0.0, 1.0, size=(mlp.depth, mlp.width)).astype(fnp.float32)
        )

    def teardown(self) -> None:
        self._context = None
        self._setup_rng = None


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)  # phase-1 competition shape (warmup round used depth=8)
    compare_against_monte_carlo(Estimator(), mlp)

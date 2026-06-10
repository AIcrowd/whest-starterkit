"""Ship a precomputed weight file and load it in setup().

Prepare `weights.npz` offline (plain numpy is fine), keep it next to your
estimator, and `whest package` bundles it automatically (folder submission).
`whest validate` / `whest run` set `context.submission_dir` to your estimator's
folder, so the exact same load path works locally and on the grader.
Loading is pickle-free and costs 0 FLOPs.
"""

from __future__ import annotations

from pathlib import Path

import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext


class Estimator(BaseEstimator):
    def setup(self, context: SetupContext) -> None:
        scale = None
        if context.submission_dir is not None:
            weights_path = Path(context.submission_dir) / "weights.npz"
            if weights_path.exists():
                scale = fnp.load(weights_path)["scale"]  # 0 FLOPs to load
        # Fallback keeps the example runnable without the weight file.
        self._scale = scale if scale is not None else fnp.ones(())

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        return fnp.zeros((mlp.depth, mlp.width)) * self._scale


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=8, seed=0)
    estimator = Estimator()
    # The framework always calls setup() before predict(); do the same here.
    # submission_dir points at this folder, so a weights.npz next to this file
    # is picked up exactly as it would be inside a packaged submission.
    estimator.setup(
        SetupContext(
            width=256,
            depth=8,
            flop_budget=68_000_000_000,
            api_version="1.0",
            submission_dir=str(Path(__file__).resolve().parent),
        )
    )
    compare_against_monte_carlo(estimator, mlp)

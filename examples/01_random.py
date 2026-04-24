from __future__ import annotations

import whest as we
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP


class Estimator(BaseEstimator):
    """Random estimator: returns random predictions for all layers."""

    def __init__(self) -> None:
        self._predict_calls = 0
        self._context = None

    def setup(self, context: SetupContext) -> None:
        self._context = context
        self._predict_calls = 0

    def predict(self, mlp: MLP, budget: int) -> we.ndarray:
        self._predict_calls += 1
        seed_text = f"random|call={self._predict_calls}|w={mlp.width}|d={mlp.depth}|b={budget}"
        seed_entropy = we.frombuffer(seed_text.encode("utf-8"), dtype=we.uint8).astype(we.int32)
        rng = we.random.default_rng(seed_entropy)
        return we.array(rng.uniform(0.0, 1.0, size=(mlp.depth, mlp.width)).astype(we.float32))

    def teardown(self) -> None:
        self._context = None
        self._predict_calls = 0


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=32, depth=6, seed=0)
    compare_against_monte_carlo(Estimator(), mlp)

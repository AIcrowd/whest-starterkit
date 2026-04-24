"""Your estimator. Edit `predict()`. Run `python estimator.py` to iterate.

Stage 1 of the WhestBench ladder: just whest and the local engine. No CLI
knowledge required. Once `predict()` returns something interesting, climb to
Stage 2: `whest validate --estimator estimator.py`.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import whest as we
from whestbench import MLP, BaseEstimator


class Estimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> we.ndarray:
        # TODO: replace this all-zeros baseline with your idea.
        _ = budget
        return we.zeros((mlp.depth, mlp.width))


def _load_baseline(name: str) -> type[BaseEstimator]:
    """Load the `Estimator` class from `examples/<name>.py` or `examples/0N_<name>.py`."""
    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(
        f"\n[whest-starterkit] Could not find baseline `{name}` in examples/.\n"
        f"Available: {sorted(p.name for p in examples_dir.glob('*.py'))}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate on your estimator locally.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Compare your estimator against an example: 'random', 'mean_propagation', "
        "'covariance_propagation', or 'combined'.",
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)

    print("--- Your estimator ---")
    compare_against_monte_carlo(Estimator(), mlp)

    if args.baseline:
        baseline_cls = _load_baseline(args.baseline)
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(baseline_cls(), mlp)

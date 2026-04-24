"""Unit tests for local_engine helpers."""
from __future__ import annotations

import pytest
import whest as we
from whestbench import MLP


def test_build_mlp_returns_mlp_with_correct_shape():
    from local_engine import build_mlp

    mlp = build_mlp(width=8, depth=3, seed=0)

    assert isinstance(mlp, MLP)
    assert mlp.width == 8
    assert mlp.depth == 3
    assert len(mlp.weights) == 3
    for w in mlp.weights:
        assert w.shape == (8, 8)


def test_build_mlp_is_deterministic_given_seed():
    from local_engine import build_mlp

    a = build_mlp(width=4, depth=2, seed=42)
    b = build_mlp(width=4, depth=2, seed=42)

    for wa, wb in zip(a.weights, b.weights):
        assert float(we.max(we.abs(wa - wb))) == 0.0


def test_build_mlp_he_initialization_scale():
    """He init: weights ~ N(0, sqrt(2/width)). Variance of a 1024x1024 weight
    matrix should be close to 2/1024 = ~0.00195."""
    from local_engine import build_mlp

    mlp = build_mlp(width=1024, depth=1, seed=0)
    var = float(we.mean(mlp.weights[0] ** 2))

    expected = 2.0 / 1024
    assert var == pytest.approx(expected, rel=0.05), (
        f"variance {var} not within 5% of He target {expected}"
    )


def test_build_mlp_rejects_invalid_dimensions():
    from local_engine import build_mlp

    with pytest.raises(ValueError):
        build_mlp(width=0, depth=3, seed=0)
    with pytest.raises(ValueError):
        build_mlp(width=8, depth=0, seed=0)


def test_monte_carlo_layer_means_returns_correct_shape():
    from local_engine import build_mlp, monte_carlo_layer_means

    mlp = build_mlp(width=8, depth=3, seed=0)
    means = monte_carlo_layer_means(mlp, n_samples=100, seed=0)

    assert means.shape == (3, 8)


def test_monte_carlo_layer_means_is_deterministic():
    from local_engine import build_mlp, monte_carlo_layer_means

    mlp = build_mlp(width=4, depth=2, seed=0)
    a = monte_carlo_layer_means(mlp, n_samples=50, seed=42)
    b = monte_carlo_layer_means(mlp, n_samples=50, seed=42)

    import whest as we
    assert float(we.max(we.abs(a - b))) == 0.0


def test_compare_against_mc_preflight_rejects_wrong_shape(capsys):
    """Estimator returning the wrong shape should print a one-line diagnostic
    and SystemExit cleanly, not raise a numpy traceback."""
    import whest as we
    from whestbench import BaseEstimator

    from local_engine import build_mlp, compare_against_monte_carlo

    class WrongShapeEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> we.ndarray:
            return we.zeros((mlp.depth + 99, mlp.width))  # wrong rows

    mlp = build_mlp(width=4, depth=2, seed=0)

    with pytest.raises(SystemExit):
        compare_against_monte_carlo(WrongShapeEstimator(), mlp, sample_counts=(10,))

    out = capsys.readouterr().out
    assert "expected" in out.lower()
    assert "estimator-contract" in out


def test_compare_against_mc_preflight_rejects_wrong_dtype(capsys):
    """Estimator returning numpy array (not whest.ndarray) should be caught."""
    import numpy as np
    from whestbench import BaseEstimator

    from local_engine import build_mlp, compare_against_monte_carlo

    class NumpyEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> we.ndarray:
            return np.zeros((mlp.depth, mlp.width))  # type: ignore

    mlp = build_mlp(width=4, depth=2, seed=0)

    with pytest.raises(SystemExit):
        compare_against_monte_carlo(NumpyEstimator(), mlp, sample_counts=(10,))

    out = capsys.readouterr().out
    assert "whest.ndarray" in out or "we.ndarray" in out


def test_compare_against_mc_runs_clean_on_zeros_estimator(capsys):
    """Happy path: zeros estimator returns the right shape, MC sweep runs,
    a table is printed."""
    import whest as we
    from whestbench import BaseEstimator

    from local_engine import build_mlp, compare_against_monte_carlo

    class ZerosEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> we.ndarray:
            return we.zeros((mlp.depth, mlp.width))

    mlp = build_mlp(width=4, depth=2, seed=0)
    result = compare_against_monte_carlo(
        ZerosEstimator(), mlp, sample_counts=(10, 100)
    )

    assert result is None
    out = capsys.readouterr().out
    assert "n_samples" in out
    assert "MSE" in out
    assert "10" in out
    assert "100" in out


@pytest.mark.parametrize("name,max_mse", [
    # Update these tolerances if the curriculum table changes.
    ("examples/01_random.py", 1.0),
    ("examples/02_mean_propagation.py", 0.10),
    ("examples/03_covariance_propagation.py", 0.10),
    ("examples/04_combined.py", 0.10),
])
def test_example_mse_within_table_tolerance(name, max_mse):
    """examples/README.md advertises MSE values; CI keeps them honest."""
    import re
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, name],
        capture_output=True, text=True, check=True,
    )
    last_line = [line for line in result.stdout.splitlines() if line.strip()][-1]
    mse_match = re.search(r"(\d+\.\d+)\s*$", last_line)
    assert mse_match, f"Could not parse MSE from: {last_line}"
    mse = float(mse_match.group(1))
    assert mse <= max_mse, f"{name} MSE {mse} exceeds curriculum-table cap {max_mse}"

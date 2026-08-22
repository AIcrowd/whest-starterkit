"""Unit tests for local_engine helpers."""

from __future__ import annotations

from pathlib import Path

import flopscope.numpy as fnp
import pytest
from whestbench import MLP

# Anchor example paths to the repo root so the subprocess tests below do not
# depend on pytest's cwd (an IDE run-configuration often sets it to tests/).
REPO_ROOT = Path(__file__).resolve().parent.parent


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
        assert float(fnp.max(fnp.abs(wa - wb))) == 0.0


def test_build_mlp_he_initialization_scale():
    """He init: weights ~ N(0, 2/width) — that second parameter is the variance,
    so the std is sqrt(2/width). Variance of a 1024x1024 weight matrix should be
    close to 2/1024 = ~0.00195."""
    from local_engine import build_mlp

    mlp = build_mlp(width=1024, depth=1, seed=0)
    var = float(fnp.mean(mlp.weights[0] ** 2))

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

    assert float(fnp.max(fnp.abs(a - b))) == 0.0


def test_compare_against_mc_preflight_rejects_wrong_shape(capsys):
    """Estimator returning the wrong shape should print a one-line diagnostic
    and SystemExit cleanly, not raise a numpy traceback."""
    from whestbench import BaseEstimator

    from local_engine import build_mlp, compare_against_monte_carlo

    class WrongShapeEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
            return fnp.zeros((mlp.depth + 99, mlp.width))  # wrong rows

    mlp = build_mlp(width=4, depth=2, seed=0)

    with pytest.raises(SystemExit):
        compare_against_monte_carlo(WrongShapeEstimator(), mlp, sample_counts=(10,))

    out = capsys.readouterr().out
    assert "expected" in out.lower()
    assert "estimator-contract" in out


def test_compare_against_mc_preflight_rejects_wrong_dtype(capsys):
    """Estimator returning numpy array (not flopscope.numpy.ndarray) should be caught."""
    import numpy as np
    from whestbench import BaseEstimator

    from local_engine import build_mlp, compare_against_monte_carlo

    class NumpyEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
            return np.zeros((mlp.depth, mlp.width))  # type: ignore

    mlp = build_mlp(width=4, depth=2, seed=0)

    with pytest.raises(SystemExit):
        compare_against_monte_carlo(NumpyEstimator(), mlp, sample_counts=(10,))

    out = capsys.readouterr().out
    assert "flopscope.numpy.ndarray" in out or "fnp.ndarray" in out


def test_compare_against_mc_runs_clean_on_zeros_estimator(capsys):
    """Happy path: zeros estimator returns the right shape, MC sweep runs,
    a table is printed."""
    from whestbench import BaseEstimator

    from local_engine import build_mlp, compare_against_monte_carlo

    class ZerosEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
            return fnp.zeros((mlp.depth, mlp.width))

    mlp = build_mlp(width=4, depth=2, seed=0)
    result = compare_against_monte_carlo(ZerosEstimator(), mlp, sample_counts=(10, 100))

    assert result is None
    out = capsys.readouterr().out
    assert "n_samples" in out
    # Both metrics are tabulated, named as the grader names them.
    assert "all_layers_mse" in out
    assert "final_layer_mse" in out
    assert "10" in out
    assert "100" in out


@pytest.mark.parametrize(
    "name,expected_all,expected_final,rel",
    [
        # The advertised n=100k values on the 1024x16 seed-0 MLP. `expected_all` is
        # the `all_layers_mse` column examples/README.md tabulates; `expected_final`
        # is the ranked `final_layer_mse` quoted in the note under that table. Bands
        # are tight on purpose: a loose cap lets an accuracy regression that leaves
        # the FLOP total untouched sail through CI.
        ("examples/01_random.py", 0.5246, 0.8432, 0.02),
        ("examples/02_mean_propagation.py", 0.000171, 0.000300, 0.05),
        ("examples/03_covariance_propagation.py", 0.0000039, 0.0000042, 0.10),
    ],
)
def test_example_mse_within_table_tolerance(name, expected_all, expected_final, rel):
    """examples/README.md advertises MSE values; CI keeps them honest."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / name)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    last_line = [line for line in result.stdout.splitlines() if line.strip()][-1]
    # Parse by column rather than by trailing float: the table ends with
    # final_layer_mse today, and pinning both columns explicitly means a future
    # column reorder fails loudly instead of silently testing the wrong number.
    cells = [c.strip() for c in last_line.split("|")]
    assert len(cells) == 5, f"Unexpected table shape: {last_line}"
    all_layers_mse, final_layer_mse = float(cells[3]), float(cells[4])

    assert all_layers_mse == pytest.approx(expected_all, rel=rel), (
        f"{name} all_layers_mse {all_layers_mse} left the examples/README.md "
        f"band {expected_all}+-{rel:.0%}"
    )
    assert final_layer_mse == pytest.approx(expected_final, rel=rel), (
        f"{name} final_layer_mse {final_layer_mse} left the examples/README.md "
        f"band {expected_final}+-{rel:.0%}"
    )

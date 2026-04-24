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

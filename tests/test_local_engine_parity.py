"""Parity test: local_engine.build_mlp must produce statistically equivalent
MLPs to whestbench.sample_mlp. Catches pedagogical drift from upstream."""
from __future__ import annotations

import pytest
import whest as we
from whestbench import sample_mlp

from local_engine import build_mlp


def test_build_mlp_matches_whestbench_sample_mlp_distribution():
    """Both should produce He-initialized weights with comparable statistics."""
    width, depth = 256, 3

    local = build_mlp(width=width, depth=depth, seed=0)
    # whestbench.sample_mlp takes `rng` (a Generator), not `seed` — bridge here.
    upstream = sample_mlp(width=width, depth=depth, rng=we.random.default_rng(0))

    assert local.width == upstream.width
    assert local.depth == upstream.depth
    assert len(local.weights) == len(upstream.weights)

    for layer_idx, (lw, uw) in enumerate(zip(local.weights, upstream.weights)):
        assert lw.shape == uw.shape, f"layer {layer_idx} shape mismatch"

        local_var = float(we.mean(lw ** 2))
        upstream_var = float(we.mean(uw ** 2))
        assert local_var == pytest.approx(upstream_var, rel=0.10), (
            f"layer {layer_idx}: variance {local_var} drifts >10% from "
            f"whestbench.sample_mlp's {upstream_var}"
        )

        local_mean = float(we.mean(lw))
        upstream_mean = float(we.mean(uw))
        assert abs(local_mean - upstream_mean) < 0.01, (
            f"layer {layer_idx}: mean drift exceeds 0.01"
        )

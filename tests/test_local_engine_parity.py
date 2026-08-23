"""Parity test: local_engine.build_mlp must produce statistically equivalent
MLPs to whestbench.sample_mlp. Catches pedagogical drift from upstream."""

from __future__ import annotations

import flopscope.numpy as fnp
import pytest
from whestbench import sample_mlp

from local_engine import build_mlp


def test_build_mlp_matches_whestbench_sample_mlp_distribution():
    """Both should produce He-initialized weights with comparable statistics."""
    width, depth = 256, 3

    local = build_mlp(width=width, depth=depth, seed=0)
    # sample_mlp takes both: `rng` drives weight sampling, `seed` only stamps
    # MLP.seed for the estimator to read. Pass rng here — `seed=` alone would
    # leave the weights unseeded.
    upstream = sample_mlp(width=width, depth=depth, rng=fnp.random.default_rng(0))

    assert local.width == upstream.width
    assert local.depth == upstream.depth
    assert len(local.weights) == len(upstream.weights)

    for layer_idx, (lw, uw) in enumerate(zip(local.weights, upstream.weights)):
        assert lw.shape == uw.shape, f"layer {layer_idx} shape mismatch"

        local_var = float(fnp.mean(lw**2))
        upstream_var = float(fnp.mean(uw**2))
        assert local_var == pytest.approx(upstream_var, rel=0.10), (
            f"layer {layer_idx}: variance {local_var} drifts >10% from "
            f"whestbench.sample_mlp's {upstream_var}"
        )

        local_mean = float(fnp.mean(lw))
        upstream_mean = float(fnp.mean(uw))
        assert abs(local_mean - upstream_mean) < 0.01, f"layer {layer_idx}: mean drift exceeds 0.01"


def test_sample_mlp_seed_stamps_metadata_without_driving_weights():
    """Pin the `rng` / `seed` distinction the comment above relies on: `seed` only
    stamps MLP.seed, and on its own leaves weight sampling unseeded."""
    assert sample_mlp(8, 2, rng=fnp.random.default_rng(0), seed=99).seed == 99

    a = sample_mlp(8, 2, seed=99)
    b = sample_mlp(8, 2, seed=99)
    assert float(fnp.max(fnp.abs(a.weights[0] - b.weights[0]))) != 0.0, (
        "seed= alone should not make weights reproducible — that is rng='s job"
    )

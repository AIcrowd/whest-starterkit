"""Drift gate: pin the flopscope billing facts that docs/reference/flopscope-primer.md
and docs/reference/code-patterns.md state in prose.

If a flopscope upgrade breaks one of these, the corresponding doc sentence is
stale — fix both together. Numbers verified against flopscope 0.10.0.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
import pytest


def _cost(fn) -> int:
    with flops.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
        fn()
    return ctx.flops_used


V32 = fnp.ones(1000, dtype=fnp.float32)
V64 = fnp.ones(1000, dtype=fnp.float64)


def test_float64_bills_double_float32():
    """Primer: 'float64 work bills 2x float32' (dtype_rate)."""
    assert _cost(lambda: V32 + V32) == 1000
    assert _cost(lambda: V64 + V64) == 2000
    assert _cost(lambda: fnp.sqrt(V64)) == 2 * _cost(lambda: fnp.sqrt(V32))


def test_transcendentals_bill_16x_but_sqrt_does_not():
    """Primer: exp/log are 16-tier; sqrt stays 1-tier."""
    assert _cost(lambda: fnp.exp(V32)) == 16 * _cost(lambda: V32 + V32)
    assert _cost(lambda: fnp.log(V32)) == 16 * _cost(lambda: V32 + V32)
    assert _cost(lambda: fnp.sqrt(V32)) == _cost(lambda: V32 + V32)


def test_pow2_bills_16x_elementwise_multiply():
    """code-patterns: write x * x, not x ** 2."""
    assert _cost(lambda: V32**2) == 16 * _cost(lambda: V32 * V32)


def test_sort_bills_comparison_count_not_flat_4x():
    """Primer/code-patterns: sorts bill ≈4·N·ceil(log2 N) (per comparison), not flat 4×N."""
    assert _cost(lambda: fnp.sort(V32)) == 4 * 1000 * 10  # N=1000, ceil(log2 N)=10


def test_rng_draw_dtype_pricing():
    """Primer: standard_normal is 16/elem at float32, 32/elem at float64 (default)."""
    rng = fnp.random.default_rng(0)
    assert _cost(lambda: rng.standard_normal(1000, dtype=fnp.float32)) == 16_000
    assert _cost(lambda: rng.standard_normal(1000)) == 32_000


def test_zeros_and_views_free_fills_and_movement_billed():
    """Primer free-list: zeros/views free; ones/stack/reshape/copy billed."""
    assert _cost(lambda: fnp.zeros((32, 256))) == 0
    assert _cost(lambda: V32.T) == 0
    assert _cost(lambda: fnp.ones((32, 256), dtype=fnp.float32)) == 32 * 256
    assert _cost(lambda: fnp.stack([fnp.zeros(256, dtype=fnp.float32)] * 4)) == 4 * 256
    assert _cost(lambda: V32.reshape(50, 20)) == 1000
    assert _cost(lambda: V32.copy()) == 1000


def test_deduct_requires_dtypes_and_applies_rate():
    """Primer: deduct() requires dtypes= (pass () for dtype-neutral); dtype rate applies.

    Real 0.10.0 signature: deduct(op_name, *, flop_cost, subscripts, shapes,
    dtypes, complex_factor_override=None) — verified 2026-07-31.
    """
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as ctx:
        with pytest.raises(TypeError):
            ctx.deduct("bad_op", flop_cost=10, subscripts=None, shapes=(), dtypes=None)
        before = ctx.flops_used
        ctx.deduct("neutral_op", flop_cost=10, subscripts=None, shapes=(), dtypes=())
        assert ctx.flops_used == before + 10
        before = ctx.flops_used
        ctx.deduct("f64_op", flop_cost=10, subscripts=None, shapes=(), dtypes=(fnp.float64,))
        assert ctx.flops_used == before + 20


def test_unsupported_dtype_error_exists_and_is_type_error():
    """Troubleshooting: UnsupportedDtypeError is catchable as TypeError."""
    assert issubclass(flops.errors.UnsupportedDtypeError, TypeError)


def test_matmul_bills_mults_and_adds():
    """Primer: matmul bills M*N*(2K-1) at float32 (random operands; structure discounts excluded)."""
    rng = fnp.random.default_rng(0)
    a = fnp.array(rng.standard_normal((10, 256), dtype=fnp.float32))
    b = fnp.array(rng.standard_normal((256, 256), dtype=fnp.float32))
    assert _cost(lambda: a @ b) == 10 * 256 * (2 * 256 - 1)


def test_example_predict_totals_pinned():
    """manage-flop-budget/algorithm-ideas/scoring-model/problem-setup cite these exact totals."""
    import importlib.util
    from pathlib import Path

    from local_engine import build_mlp

    mlp = build_mlp(width=256, depth=32, seed=0)
    totals = {}
    for name, expected in [
        ("02_mean_propagation", 20_333_056),
        ("03_covariance_propagation", 3_262_575_998),
    ]:
        path = Path(__file__).resolve().parent.parent / "examples" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        est = mod.Estimator()
        with flops.BudgetContext(flop_budget=10**13, quiet=True) as ctx:
            est.predict(mlp, 10**13)
        totals[name] = ctx.flops_used
        assert ctx.flops_used == expected, f"{name}: {ctx.flops_used:,} != {expected:,}"

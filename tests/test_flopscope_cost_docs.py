"""Drift gate: pin the flopscope billing facts that docs/reference/flopscope-primer.md
and docs/reference/code-patterns.md state in prose.

If a flopscope upgrade breaks one of these, the corresponding doc sentence is
stale — fix both together. Numbers verified against flopscope 0.9.1.
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

    Real 0.9.1 signature: deduct(op_name, *, flop_cost, subscripts, shapes,
    dtypes, complex_factor_override=None) — verified 2026-07-24.
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

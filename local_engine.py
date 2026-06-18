"""local_engine.py

Pedagogical re-implementation of whestbench primitives using flopscope's
NumPy-shaped API (``flopscope.numpy``).

This module deliberately stays single-file and uses just the canonical
flopscope idiom — ``import flopscope as flops; import flopscope.numpy as
fnp`` — see ``docs/reference/code-patterns.md``. Most operations are
``fnp.*`` calls; ``flops`` is only reached for ``BudgetContext``. We keep
the surface intentionally narrow here for clarity, mirroring how
participants will mostly write their code.

Drift from whestbench is detected by ``tests/test_local_engine_parity.py``.
Do NOT refactor this file to ``from whestbench import ...`` — see Issue #1
in the design spec for rationale.
"""

from __future__ import annotations

import sys
from pathlib import Path

# IDE "Run File" friendliness: ensure the repo root is on sys.path so that
# `from local_engine import ...` works whether the script is run from the
# repo root or from an IDE that sets cwd to the file's directory.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Friendly import guard: surface a clear "run `uv sync`" message instead of a
# bare ImportError traceback if dependencies are missing.
try:
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import MLP, BaseEstimator
except ImportError as exc:  # pragma: no cover — exercised manually
    raise SystemExit(
        "\n[whest-starterkit] Could not import `flopscope` / `whestbench`.\n"
        "Run `uv sync` from the repo root, then re-run this script.\n"
        f"(Original error: {exc})\n"
    ) from exc


def build_mlp(width: int, depth: int, seed: int = 0) -> MLP:
    """Return a square MLP with He-initialized N(0, 2/width) weights.

    Deterministic given `seed`. Uses raw flopscope primitives only.
    """
    if width < 1 or depth < 1:
        raise ValueError(f"build_mlp requires width>=1 and depth>=1; got {width=}, {depth=}")
    rng = fnp.random.default_rng(seed)
    scale = (2.0 / width) ** 0.5
    weights = [
        fnp.array((rng.standard_normal((width, width)) * scale).astype(fnp.float32))
        for _ in range(depth)
    ]
    return MLP(width=width, depth=depth, weights=weights)


def monte_carlo_layer_means(
    mlp: MLP,
    n_samples: int,
    seed: int = 0,
) -> fnp.ndarray:
    """Forward `n_samples` N(0,1) inputs through `mlp.weights` and average per layer.

    Returns shape `(depth, width)` — same shape as `Estimator.predict` so the two
    can be subtracted directly.
    """
    rng = fnp.random.default_rng(seed)
    width = mlp.width
    x = fnp.array(rng.standard_normal((n_samples, width)).astype(fnp.float32))
    rows = []
    for w in mlp.weights:
        x = fnp.maximum(fnp.matmul(x, w), 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def compare_against_monte_carlo(
    estimator: BaseEstimator,
    mlp: MLP,
    sample_counts: tuple[int, ...] = (10, 100, 1_000, 10_000, 100_000),
    estimator_budget: int = int(4e9),
    sampling_budget: int = int(1e12),
    seed: int = 0,
) -> None:
    """Run estimator once, then sweep MC at each sample count and print a table.

    Friendly preflight: before the MC sweep, validate the estimator returns the
    right shape/dtype on the actual MLP. On failure, print a one-line diagnostic
    pointing at the contract doc and exit cleanly (SystemExit) — no numpy traceback.

    Returns None — this is a print helper for stage-1 dev loops.
    """
    expected_shape = (mlp.depth, mlp.width)

    try:
        with flops.BudgetContext(flop_budget=estimator_budget, quiet=True) as est_ctx:
            est_pred = estimator.predict(mlp, estimator_budget)
    except Exception as exc:
        import inspect

        try:
            src_file = inspect.getsourcefile(type(estimator))
        except TypeError:
            src_file = "<unknown>"
        print(
            f"\n[whest-starterkit] Your estimator raised at {src_file} "
            f"during predict():\n  {type(exc).__name__}: {exc}\n"
            f"See docs/reference/estimator-contract.md\n"
        )
        raise SystemExit(2) from exc

    if not isinstance(est_pred, fnp.ndarray):
        print(
            f"\n[whest-starterkit] predict() must return a `flopscope.numpy.ndarray`, "
            f"got `{type(est_pred).__name__}`.\n"
            f"Tip: use `import flopscope.numpy as fnp` and return `fnp.zeros(...)` or "
            f"`fnp.array(...)`.\n"
            f"See docs/reference/estimator-contract.md\n"
        )
        raise SystemExit(2)

    if est_pred.shape != expected_shape:
        print(
            f"\n[whest-starterkit] predict() returned shape {tuple(est_pred.shape)}, "
            f"expected (depth={mlp.depth}, width={mlp.width}).\n"
            f"See docs/reference/estimator-contract.md\n"
        )
        raise SystemExit(2)

    estimator_flops = est_ctx.flops_used

    row = "{:>10} | {:>14} | {:>15} | {:>10}".format
    header = row("n_samples", "sampling_flops", "estimator_flops", "MSE")
    print(f"MLP: width={mlp.width} depth={mlp.depth} seed={seed}\n")
    print(header)
    print("-" * len(header))
    for n in sample_counts:
        with flops.BudgetContext(flop_budget=sampling_budget, quiet=True) as mc_ctx:
            sampled = monte_carlo_layer_means(mlp, n, seed=seed)
        mse = float(fnp.mean((est_pred - sampled) ** 2))
        print(row(f"{n:,}", f"{mc_ctx.flops_used:,}", f"{estimator_flops:,}", f"{mse:.6f}"))

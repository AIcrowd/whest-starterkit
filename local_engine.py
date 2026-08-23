"""local_engine.py

Pedagogical re-implementation of whestbench primitives using flopscope's
NumPy-shaped API (``flopscope.numpy``).

This module deliberately stays single-file and uses the canonical flopscope
idiom, ``import flopscope as flops; import flopscope.numpy as fnp`` (see
``docs/reference/code-patterns.md``). Most operations are ``fnp.*`` calls;
``flops`` is only reached for ``BudgetContext``. That narrow surface matches
how most participants write their own code.

``tests/test_local_engine_parity.py`` detects drift from whestbench.
The ``MLP`` / ``BaseEstimator`` imports below are deliberate; they are the
participant-facing types. What must NOT be imported is whestbench's own
engine (``sample_mlp``, ``sample_layer_statistics``): this file exists to
re-implement those in ``fnp``.
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

# Import guard: if a dependency is missing, surface a clear "run `uv sync`"
# message instead of a bare ImportError traceback.
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
    return MLP(width=width, depth=depth, weights=weights, seed=seed)


def monte_carlo_layer_means(
    mlp: MLP,
    n_samples: int,
    seed: int = 0,
) -> fnp.ndarray:
    """Forward `n_samples` N(0,1) float32 inputs through `mlp.weights` and average per layer.

    Draws in float32 and accumulates the mean in float64, matching
    whestbench's own ground-truth sampler. flopscope also bills float64 ops at
    2x, so the float32 draw costs less.

    Returns shape `(depth, width)`, the same shape as `Estimator.predict`, so you
    can subtract the two directly.
    """
    rng = fnp.random.default_rng(seed)
    width = mlp.width
    x = fnp.array(rng.standard_normal((n_samples, width), dtype=fnp.float32))
    rows = []
    for w in mlp.weights:
        x = fnp.maximum(fnp.matmul(x, w), 0.0)
        rows.append(fnp.mean(fnp.asarray(x, dtype=fnp.float64), axis=0))
    return fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)


def compare_against_monte_carlo(
    estimator: BaseEstimator,
    mlp: MLP,
    sample_counts: tuple[int, ...] = (10, 100, 1_000, 10_000, 100_000),
    estimator_budget: int = 2**41,
    sampling_budget: int = int(5e12),
    seed: int = 0,
    submission_dir: str | None = None,
) -> None:
    """Run estimator once, then sweep MC at each sample count and print a table.

    `estimator_budget` defaults to the competition per-MLP budget, 2**41 FLOPs.
    Your `predict()` is metered against the same 2**41 FLOP budget and the same
    120 s per-predict wall cap the grader uses, and the 0.4 s residual-time gate
    is checked after the run. Not modelled here: the 5 s `setup()` timeout, the
    8 GB memory limit, and the grader's zero-and-continue behaviour on failure
    (this helper exits instead). Use `whest run` for those.

    `sampling_budget` caps one Monte-Carlo row, not the whole sweep: each sample
    count runs in its own `BudgetContext`, so the ceiling only has to clear the
    largest single row (n=100,000 bills 3,363,737,665,536 FLOPs at width 1024,
    depth 16). The 5e12 default leaves headroom above it. Ground-truth sampling
    is a local convenience. It is not charged against your submission, so its
    context deliberately carries no wall cap.

    `final_layer_mse` is the ranked metric. The grader scores
    `adjusted_final_layer_score = final_layer_mse * max(0.1, C_m / B_m)`.
    `all_layers_mse` is whestbench's secondary metric; it is the more forgiving
    of the two, so do not tune against it alone.

    Preflight: before the MC sweep, validate the estimator returns an
    `fnp.ndarray` of the right shape on the actual MLP (dtype is not checked
    here; the grader casts with `fnp.asarray(..., dtype=fnp.float32)`). On
    failure, print a one-line diagnostic pointing at the contract doc, then exit
    cleanly with SystemExit instead of a numpy traceback.

    Returns None; this is a print helper for stage-1 dev loops.
    """
    expected_shape = (mlp.depth, mlp.width)

    import inspect

    from whestbench import SetupContext

    # `submission_dir` is where the grader puts your packaged folder, and it is how
    # the shipped-weights pattern (examples/04) finds its `.npz`. Default it to the
    # directory the estimator class was defined in, which is what the grader does
    # for a folder submission, so an estimator.py and weights.npz sitting side by
    # side resolve with no extra configuration.
    # Pass `submission_dir=` explicitly when the weights live elsewhere
    # (examples/04 saves into a tempdir rather than committing an .npz to the repo).
    try:
        _src = inspect.getsourcefile(type(estimator))
    except TypeError:  # C/builtin type — no source file to locate
        _src = None
    _submission_dir = submission_dir or (
        str(Path(_src).resolve().parent) if _src else str(_REPO_ROOT)
    )

    # setup() runs OUTSIDE the BudgetContext, matching the grader: it calls
    # setup in `runner.start` (runner.py:175), not inside the per-predict budget.
    # Any FLOPs you spend here are free, but the grader does hold it to a 5 s
    # timeout that this helper does not model.
    estimator.setup(
        SetupContext(
            width=mlp.width,
            depth=mlp.depth,
            flop_budget=estimator_budget,
            api_version="1.0",
            submission_dir=_submission_dir,
            seed=seed,
        )
    )

    # `wall_time_limit_s` mirrors the grader's per-predict cap (whestbench 0.16.0
    # scoring.py:709-712 passes spec.wall_time_limit_s = 120.0). Without it a slow
    # estimator prints a clean table locally and still fails on the grader.
    try:
        with flops.BudgetContext(
            flop_budget=estimator_budget, wall_time_limit_s=120.0, quiet=True
        ) as est_ctx:
            est_pred = estimator.predict(mlp, estimator_budget)
    except Exception as exc:
        src_file = _src or "<unknown>"
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

    # The residual-time gate: wall time inside the context that flopscope could
    # not attribute to a metered op. The grader fails the MLP outright when this
    # exceeds 0.4 s (multiplier forced to 1.0), so surface it before the table.
    if est_ctx.residual_wall_time_s > 0.4:
        print(
            f"[whest-starterkit] RESIDUAL GATE: {est_ctx.residual_wall_time_s:.3f}s > 0.400s "
            f"— the grader would FAIL this MLP (multiplier forced to 1.0).\n"
        )

    # `final_layer_mse` goes last on purpose: it is the ranked metric, so it ends
    # the row that reports the result. tests/test_local_engine.py pins both MSE
    # columns by position, so reordering this row fails CI rather than silently
    # moving what is tested. sampling_flops is 17 chars wide at n=100,000
    # (3,363,737,665,536).
    row = "{:>10} | {:>17} | {:>15} | {:>14} | {:>15}".format
    header = row(
        "n_samples", "sampling_flops", "estimator_flops", "all_layers_mse", "final_layer_mse"
    )
    print(
        f"MLP: width={mlp.width} depth={mlp.depth} seed={mlp.seed}  "
        f"(MC sampling seed={seed})\n"
    )
    print(header)
    print("-" * len(header))
    for n in sample_counts:
        with flops.BudgetContext(flop_budget=sampling_budget, quiet=True) as mc_ctx:
            sampled = monte_carlo_layer_means(mlp, n, seed=seed)
        diff = est_pred - sampled
        all_layers_mse = float(fnp.mean(diff * diff))
        final_layer_mse = float(fnp.mean((est_pred[-1] - sampled[-1]) ** 2))
        print(
            row(
                f"{n:,}",
                f"{mc_ctx.flops_used:,}",
                f"{estimator_flops:,}",
                f"{all_layers_mse:.6f}",
                f"{final_layer_mse:.6f}",
            )
        )

    # Mirror the grader's lifecycle end-to-end: setup() above, teardown() here.
    estimator.teardown()

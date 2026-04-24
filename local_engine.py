# local_engine.py
#
# Pedagogical re-implementation of whestbench primitives in raw `whest` code.
# Kept verbose on purpose so participants can read the full forward pass and
# understand exactly what whestbench does for them under the hood.
# Drift from whestbench is detected by tests/test_local_engine_parity.py.
# Do NOT refactor this file to `from whestbench import ...` — see Issue #1
# in the design spec for rationale.

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
# bare ImportError traceback if whestbench is not installed.
try:
    import whest as we
    from whestbench import MLP, BaseEstimator
except ImportError as exc:  # pragma: no cover — exercised manually
    raise SystemExit(
        "\n[whest-starterkit] Could not import `whest` / `whestbench`.\n"
        "Run `uv sync` from the repo root, then re-run this script.\n"
        f"(Original error: {exc})\n"
    ) from exc


def build_mlp(width: int, depth: int, seed: int = 0) -> MLP:
    """Return a square MLP with He-initialized N(0, 2/width) weights.

    Deterministic given `seed`. Uses raw whest primitives only.
    """
    if width < 1 or depth < 1:
        raise ValueError(f"build_mlp requires width>=1 and depth>=1; got {width=}, {depth=}")
    rng = we.random.default_rng(seed)
    scale = (2.0 / width) ** 0.5
    weights = [
        we.array((rng.standard_normal((width, width)) * scale).astype(we.float32))
        for _ in range(depth)
    ]
    return MLP(width=width, depth=depth, weights=weights)

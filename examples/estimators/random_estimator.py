from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from circuit_estimation import BaseEstimator, SetupContext
from circuit_estimation.domain import Circuit


class Estimator(BaseEstimator):
    """A friendly first estimator: intentionally silly, fully correct interface.

    Welcome! This file is designed to be the *first* estimator you read.
    The model behavior is intentionally goofy: for each depth, it emits a
    random vector in ``[-1, 1]`` and calls that the prediction. It is not meant
    to be accurate. Its job is to teach the contract with minimal math baggage.

    What this class demonstrates end-to-end:

    1. ``setup(context)`` (optional): one-time initialization before scoring.
    2. ``predict(circuit, budget)`` (required): stream exactly one ``(n,)``
       row per depth with ``yield``.
    3. ``teardown()`` (optional): cleanup after all scoring calls finish.

    The core output contract is the thing to memorize:
    - input: one ``Circuit`` + one integer ``budget``
    - output: streamed rows, one per depth
    - each row shape: ``(circuit.n,)``
    - row dtype: ``np.float32``
    - row count: exactly ``circuit.d`` yields

    Gotchas that bite first-time implementations:
    - Returning one final ``(depth, width)`` tensor is wrong. You must stream.
    - Emitting rows with shape ``(1, n)`` or ``(n, 1)`` is wrong.
    - Emitting Python lists works sometimes locally, but ndarray float32 is
      the safe/expected format.
    - Forgetting to use ``budget`` can be fine in toy code, but real solutions
      typically switch behavior by budget.
    - Reusing one mutable array object and mutating it *after* yielding can
      produce confusing bugs in custom pipelines.

    For context, real estimators aim to approximate quantities like ``E[x]``.
    This tutorial estimator does none of that, so score quality will be poor.
    Its value is pedagogical: clean shape/count semantics, clear lifecycle
    hooks, and a simple baseline structure you can evolve.

    Complexity is tiny: ``O(d * n)`` time and ``O(n)`` transient memory.
    """

    def __init__(self) -> None:
        self._seed_prefix = "random-estimator"
        self._predict_calls = 0
        self._context: SetupContext | None = None

    def setup(self, context: SetupContext) -> None:
        """Initialize one-time state before ``predict`` is called.

        ``context`` contains static evaluation metadata (width/depth/budgets,
        tolerance, and API version). You can cache expensive constants here.
        In this toy estimator, we build a readable seed prefix from context so
        runs are reproducible enough for debugging.
        """
        self._context = context
        budgets = ",".join(str(value) for value in context.budgets)
        self._seed_prefix = (
            "random-estimator"
            f"|api={context.api_version}"
            f"|width={context.width}"
            f"|max_depth={context.max_depth}"
            f"|budgets={budgets}"
            f"|tolerance={context.time_tolerance:.6f}"
        )
        self._predict_calls = 0

    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        """Yield one random prediction row per depth.

        Even though this estimator is silly, the interface behavior is exactly
        what the scorer expects:
        - one row emitted per depth,
        - each row shape ``(circuit.n,)``,
        - float32 numeric values.

        We use a deterministic per-call seed string mixed from context, budget,
        and call index. That keeps behavior random-looking while debuggable.
        """
        self._predict_calls += 1
        seed_text = (
            f"{self._seed_prefix}"
            f"|call={self._predict_calls}"
            f"|n={circuit.n}"
            f"|d={circuit.d}"
            f"|budget={max(budget, 0)}"
        )
        seed_entropy = np.frombuffer(seed_text.encode("utf-8"), dtype=np.uint8).astype(np.uint32)
        rng = np.random.default_rng(seed_entropy)

        for _depth in range(circuit.d):
            # Uniform random row in [-1, 1], cast to the expected float32 dtype.
            row = np.asarray(
                rng.uniform(-1.0, 1.0, size=(circuit.n,)),
                dtype=np.float32,
            )
            yield row

    def teardown(self) -> None:
        """Release process-level resources after scoring ends.

        This hook is optional, but useful when your estimator opens files,
        caches large arrays, or creates external clients. Here we only reset
        internal references.
        """
        self._context = None
        self._predict_calls = 0

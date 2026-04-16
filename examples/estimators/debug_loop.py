"""Self-contained debug loop for the WhestBench challenge — runnable two ways.

1. **Direct**: ``python examples/estimators/debug_loop.py``
   Builds one small MLP, runs the inline mean-propagation ``Estimator`` once,
   then sweeps ``SAMPLE_COUNTS`` of Monte Carlo samples and prints a
   convergence table. No CLI, no dataset, no subprocess — just a script.

2. **Scored via whestbench**: ``whest run --estimator examples/estimators/debug_loop.py``
   The inline class is named ``Estimator`` and honours the ``BaseEstimator``
   contract, so the whest CLI picks it up as a valid submission and scores
   it against the real contest spec.

Reading guide: docstring → imports → knobs → ``Estimator`` class →
hand-rolled MLP + Monte Carlo helpers → ``__main__`` table loop.
Everything uses raw whest primitives (``we.matmul``, ``we.maximum``,
``we.mean``, ``we.BudgetContext``) — no magic from ``whestbench.*``.

The table shows the central phenomenon of the challenge: the structural
estimator is nearly free in FLOPs, sampling cost grows linearly with
``n_samples``, and the MSE between the two shrinks as ``1/sqrt(n)`` before
plateauing at the estimator's intrinsic bias.
"""

from __future__ import annotations

import whest as we

from whestbench import MLP, BaseEstimator

# Tweak these freely — the direct run still finishes in under two seconds.
WIDTH = 32
DEPTH = 6
SEED = 0
SAMPLE_COUNTS = (10, 100, 1_000, 10_000, 100_000)
ESTIMATOR_BUDGET = int(1e9)
SAMPLING_BUDGET = int(1e12)


class Estimator(BaseEstimator):
    """Inline mean-propagation estimator — cheap, biased, deterministic.

    For a ReLU unit ``z = max(0, w^T x)`` with a Gaussian pre-activation
    ``pre ~ N(mu_pre, sigma_pre^2)``::

        E[z]   = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
        E[z^2] = (mu_pre^2 + sigma_pre^2) * Phi(alpha)
                 + mu_pre * sigma_pre * phi(alpha)
        Var[z] = E[z^2] - E[z]^2

    with ``alpha = mu_pre / sigma_pre``. We propagate the mean and a
    *diagonal* variance (one scalar per neuron, ignoring off-diagonal
    correlations) through every layer and return the post-ReLU mean for
    each layer stacked into a ``(depth, width)`` array.
    """

    def predict(self, mlp: MLP, budget: int) -> we.ndarray:
        _ = budget  # structural estimator cost is fixed and tiny
        width = mlp.width

        mu = we.zeros(width)
        var = we.ones(width)

        rows = []
        for w in mlp.weights:
            mu_pre = w.T @ mu
            var_pre = we.maximum((w * w).T @ var, 1e-12)
            sigma_pre = we.sqrt(var_pre)

            alpha = mu_pre / sigma_pre
            phi_alpha = we.stats.norm.pdf(alpha)
            Phi_alpha = we.stats.norm.cdf(alpha)

            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var = we.maximum(ez2 - mu * mu, 0.0)

            rows.append(mu)

        return we.stack(rows, axis=0)


def build_mlp(width: int, depth: int, rng: we.random.Generator) -> MLP:
    """He-initialise a square MLP using raw whest primitives."""
    scale = (2.0 / width) ** 0.5
    weights = [
        we.array((rng.standard_normal((width, width)) * scale).astype(we.float32))
        for _ in range(depth)
    ]
    return MLP(width=width, depth=depth, weights=weights)


def monte_carlo_layer_means(
    weights: list[we.ndarray],
    n_samples: int,
    rng: we.random.Generator,
) -> we.ndarray:
    """Forward ``n_samples`` N(0,1) inputs through ``weights`` and average per layer.

    Returns shape ``(depth, width)`` — same shape as ``Estimator.predict`` so
    the two can be subtracted directly.
    """
    width = int(weights[0].shape[0])
    x = we.array(rng.standard_normal((n_samples, width)).astype(we.float32))
    rows = []
    for w in weights:
        x = we.maximum(we.matmul(x, w), 0.0)
        rows.append(we.mean(x, axis=0))
    return we.stack(rows, axis=0)


if __name__ == "__main__":
    master = we.random.default_rng(SEED)
    mlp = build_mlp(WIDTH, DEPTH, master)

    with we.BudgetContext(flop_budget=ESTIMATOR_BUDGET, quiet=True) as est_ctx:
        est_pred = Estimator().predict(mlp, ESTIMATOR_BUDGET)
    estimator_flops = est_ctx.flops_used

    row = "{:>10} | {:>14} | {:>15} | {:>10}".format
    header = row("n_samples", "sampling_flops", "estimator_flops", "MSE")
    print(f"MLP: width={WIDTH} depth={DEPTH} seed={SEED}\n")
    print(header)
    print("-" * len(header))
    for n in SAMPLE_COUNTS:
        with we.BudgetContext(flop_budget=SAMPLING_BUDGET, quiet=True) as mc_ctx:
            sampled = monte_carlo_layer_means(mlp.weights, n, master)
        mse = float(we.mean((est_pred - sampled) ** 2))
        print(row(f"{n:,}", f"{mc_ctx.flops_used:,}", f"{estimator_flops:,}", f"{mse:.6f}"))
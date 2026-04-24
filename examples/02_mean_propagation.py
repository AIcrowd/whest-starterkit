"""Mean propagation estimator for ReLU MLPs — self-contained educational implementation.

For a ReLU unit  z = max(0, w^T x),  if the pre-activation is Gaussian:

    pre ~ N(mu_pre, sigma_pre^2)

then the exact first two moments of z are:

    E[z]   = mu_pre  * Phi(alpha) + sigma_pre * phi(alpha)
    E[z^2] = (mu_pre^2 + sigma_pre^2) * Phi(alpha) + mu_pre * sigma_pre * phi(alpha)
    Var[z] = E[z^2] - E[z]^2

where  alpha = mu_pre / sigma_pre,  phi is the standard normal PDF,
and Phi is the standard normal CDF.

This estimator propagates the mean and a *diagonal* variance (one scalar per
neuron, ignoring off-diagonal correlations) through every layer and returns the
post-ReLU mean for each layer stacked into a (depth, width) array.
"""

from __future__ import annotations

import whest as we

from whestbench import BaseEstimator
from whestbench.domain import MLP


class Estimator(BaseEstimator):
    """Mean propagation estimator for ReLU MLPs.

    Propagates means through each layer using the analytical ReLU expectation
    formula with a diagonal variance approximation (assumes independent neurons).
    """

    def predict(self, mlp: MLP, budget: int) -> we.ndarray:
        """Predict per-layer output means via first-moment propagation.

        Returns an array of shape (depth, width) where row i is the predicted
        mean activation vector after the i-th ReLU layer.
        """
        _ = budget  # budget is unused; this estimator has low FLOP cost
        width = mlp.width

        # --- Step 1: initialise the input distribution ---
        # Treat the network input as standard normal: mu=0, var=1 per dimension.
        mu = we.zeros(width)  # shape (width,)
        var = we.ones(width)  # shape (width,)  — diagonal of the covariance

        rows = []
        for w in mlp.weights:  # w has shape (width, width)
            # --- Step 2: propagate through the linear layer ---
            # Pre-activation mean:  mu_pre = W^T mu
            mu_pre = w.T @ mu

            # Pre-activation variance (diagonal only):
            #   var_pre[i] = sum_j  W[j,i]^2 * var[j]
            #              = (W^2)^T var
            var_pre = (w * w).T @ var

            # Clamp to avoid division by zero or negative values from rounding.
            var_pre = we.maximum(var_pre, 1e-12)
            sigma_pre = we.sqrt(var_pre)  # shape (width,)

            # --- Step 3: compute the standardised ratio alpha = mu / sigma ---
            alpha = mu_pre / sigma_pre

            # Evaluate the PDF and CDF at alpha
            phi_alpha = we.stats.norm.pdf(alpha)  # phi(alpha)
            Phi_alpha = we.stats.norm.cdf(alpha)  # Phi(alpha)

            # --- Step 4: ReLU expectation ---
            # E[ReLU(pre)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # --- Step 5: post-ReLU variance ---
            # E[ReLU(pre)^2] = (mu_pre^2 + var_pre) * Phi(alpha)
            #                  + mu_pre * sigma_pre * phi(alpha)
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            # Var[ReLU] = E[z^2] - E[z]^2  (clamped to 0 for numerical safety)
            var = we.maximum(ez2 - mu * mu, 0.0)

            # Record the post-ReLU mean for this layer
            rows.append(mu)

        # Stack all layer means into a single (depth, width) array
        return we.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=32, depth=6, seed=0)
    compare_against_monte_carlo(Estimator(), mlp)

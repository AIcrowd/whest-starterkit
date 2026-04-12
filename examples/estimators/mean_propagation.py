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

import mechestim as me

from whestbench import BaseEstimator
from whestbench.domain import MLP

# ---------------------------------------------------------------------------
# Helpers: standard normal PDF and CDF
# ---------------------------------------------------------------------------

# Abramowitz & Stegun approximation constants (formula 26.2.17)
_A1 = 0.254829592
_A2 = -0.284496736
_A3 = 1.421413741
_A4 = -1.453152027
_A5 = 1.061405429
_P = 0.3275911


def _norm_pdf(x: me.ndarray) -> me.ndarray:
    """Standard normal PDF: phi(x) = exp(-x^2 / 2) / sqrt(2*pi)."""
    return me.exp(-0.5 * x * x) / me.sqrt(2.0 * me.pi)


def _norm_cdf(x: me.ndarray) -> me.ndarray:
    """Standard normal CDF using the Abramowitz & Stegun approximation.

    Uses only basic mechestim operations (exp, abs). Accurate to ~1.5e-7.
    """
    t = 1.0 / (1.0 + _P * me.abs(x))
    poly = ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    cdf = 1.0 - poly * me.exp(-0.5 * x * x)
    return me.where(x >= 0, cdf, 1.0 - cdf)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class Estimator(BaseEstimator):
    """Mean propagation estimator for ReLU MLPs.

    Propagates means through each layer using the analytical ReLU expectation
    formula with a diagonal variance approximation (assumes independent neurons).
    """

    def predict(self, mlp: MLP, budget: int) -> me.ndarray:
        """Predict per-layer output means via first-moment propagation.

        Returns an array of shape (depth, width) where row i is the predicted
        mean activation vector after the i-th ReLU layer.
        """
        _ = budget  # budget is unused; this estimator has low FLOP cost
        width = mlp.width

        # --- Step 1: initialise the input distribution ---
        # Treat the network input as standard normal: mu=0, var=1 per dimension.
        mu = me.zeros(width)  # shape (width,)
        var = me.ones(width)  # shape (width,)  — diagonal of the covariance

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
            var_pre = me.maximum(var_pre, 1e-12)
            sigma_pre = me.sqrt(var_pre)  # shape (width,)

            # --- Step 3: compute the standardised ratio alpha = mu / sigma ---
            alpha = mu_pre / sigma_pre

            # Evaluate the PDF and CDF at alpha
            phi_alpha = _norm_pdf(alpha)  # phi(alpha)
            Phi_alpha = _norm_cdf(alpha)  # Phi(alpha)

            # --- Step 4: ReLU expectation ---
            # E[ReLU(pre)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # --- Step 5: post-ReLU variance ---
            # E[ReLU(pre)^2] = (mu_pre^2 + var_pre) * Phi(alpha)
            #                  + mu_pre * sigma_pre * phi(alpha)
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            # Var[ReLU] = E[z^2] - E[z]^2  (clamped to 0 for numerical safety)
            var = me.maximum(ez2 - mu * mu, 0.0)

            # Record the post-ReLU mean for this layer
            rows.append(mu)

        # Stack all layer means into a single (depth, width) array
        return me.stack(rows, axis=0)

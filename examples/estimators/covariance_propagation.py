"""Covariance propagation estimator for ReLU MLPs — self-contained educational implementation.

Unlike the diagonal (mean-propagation) approach, this estimator tracks the
*full* covariance matrix between neurons as the signal passes through each
linear + ReLU layer.

Linear layer update (exact):
    mu_pre  = W^T mu
    cov_pre = W^T cov W

ReLU update (approximate):
    After a ReLU the neurons become correlated in a complex way.  A tractable
    approximation is the "gain" method:

        gain[i] = Phi(alpha[i])   where alpha[i] = mu_pre[i] / sigma_pre[i]

    The off-diagonal entries of the post-ReLU covariance are scaled by the
    product of the corresponding gains:

        cov_post[i,j] ≈ gain[i] * gain[j] * cov_pre[i,j]

    and the diagonal is replaced by the exact marginal variance from the
    ReLU expectation formula:

        var_post[i] = E[z_i^2] - E[z_i]^2

Numerical stability:
    Deep networks can cause the covariance to grow very large.  Before each
    linear layer we check the maximum diagonal entry and rescale (mu, cov) if
    it exceeds a threshold, keeping a running log-scale to restore the mean in
    the original coordinates before recording it.
"""

from __future__ import annotations

import mechestim as me

from network_estimation import BaseEstimator
from network_estimation.domain import MLP

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

# If any diagonal entry of the covariance exceeds this value we rescale
# to keep the arithmetic well-behaved in float32.
_COV_RESCALE_THRESHOLD = 1e100


class Estimator(BaseEstimator):
    """Full covariance propagation estimator for ReLU MLPs.

    Tracks the full (width x width) covariance matrix through every layer.
    More accurate than mean propagation for correlated networks, but costs
    O(width^2) memory and O(width^3) FLOPs per layer.
    """

    def predict(self, mlp: MLP, budget: int) -> me.ndarray:
        """Predict per-layer output means via full covariance propagation.

        Returns an array of shape (depth, width) where row i is the predicted
        mean activation vector after the i-th ReLU layer.
        """
        _ = budget  # budget is unused by this estimator
        width = mlp.width

        # --- Step 1: initialise the input distribution ---
        # Input is modelled as standard multivariate normal: mu=0, cov=I.
        mu = me.zeros(width)  # shape (width,)
        cov = me.eye(width)  # shape (width, width)
        log_scale = 0.0  # tracks accumulated log of rescaling factor

        rows = []
        for w in mlp.weights:  # w has shape (width, width)
            # --- Step 2: overflow prevention ---
            # If the covariance has grown very large, rescale (mu, cov) by the
            # square root of the largest variance so that downstream matmuls
            # stay in a safe range.  We compensate in the recorded mean later.
            cov_diag = me.diag(cov)
            max_var_np = float(me.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                s = float(me.sqrt(max_var_np))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(me.log(s))

            # --- Step 3: propagate through the linear layer ---
            # Pre-activation mean:         mu_pre  = W^T mu
            # Pre-activation covariance:   cov_pre = W^T cov W
            mu_pre = w.T @ mu
            cov_pre = w.T @ cov @ w

            # Extract per-neuron pre-activation standard deviations from the
            # diagonal of cov_pre.
            var_pre = me.maximum(me.diag(cov_pre), 1e-12)
            sigma_pre = me.sqrt(var_pre)

            # --- Step 4: compute alpha = mu / sigma for each neuron ---
            alpha = mu_pre / sigma_pre
            phi_alpha = _norm_pdf(alpha)
            Phi_alpha = _norm_cdf(alpha)

            # --- Step 5: post-ReLU mean (exact per neuron) ---
            # E[ReLU(pre)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # --- Step 6: post-ReLU diagonal variance (exact per neuron) ---
            # E[z^2] = (mu_pre^2 + var_pre) * Phi(alpha) + mu_pre * sigma_pre * phi(alpha)
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = me.maximum(ez2 - mu * mu, 0.0)

            # --- Step 7: approximate post-ReLU covariance ---
            # gain[i] = Phi(alpha[i])  when sigma_pre[i] > 0, else 0
            sigma_np = me.asarray(sigma_pre, dtype=me.float64)
            Phi_np = me.asarray(Phi_alpha, dtype=me.float64)
            gain_np = me.where(sigma_np > 1e-12, Phi_np, 0.0)
            gain = me.array(gain_np.astype(me.float32))

            # Off-diagonal approximation:  cov_post[i,j] ≈ gain[i]*gain[j]*cov_pre[i,j]
            cov = me.multiply(me.outer(gain, gain), cov_pre)

            # Replace the diagonal with the exact marginal variances.
            me.fill_diagonal(cov, var_post)

            # --- Step 8: record mean in original (unscaled) coordinates ---
            scale_factor = float(me.exp(log_scale))
            rows.append(mu * scale_factor)

        # Stack all layer means into a single (depth, width) array
        return me.stack(rows, axis=0)

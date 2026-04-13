"""Budget-aware combined estimator — self-contained educational implementation.

This estimator contains *both* propagation algorithms inline and selects the
right one based on the available FLOP budget:

    budget >= 30 * width^2  →  full covariance propagation  (more accurate)
    budget <  30 * width^2  →  diagonal mean propagation    (cheaper)

The threshold comes from the observation that the covariance path costs
roughly O(width^2) extra FLOPs per layer compared to the mean path, so if
the caller has budgeted at least 30 times that per layer there is room for
the more expensive algorithm.

Math background
---------------
Both paths propagate a Gaussian approximation to the pre-activation
distribution through each linear + ReLU layer.

Linear layer  (exact in both paths):
    mu_pre  = W^T mu

ReLU layer  (uses the normal integral formula):
    alpha = mu_pre / sigma_pre
    E[ReLU(pre)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)

The two paths differ only in how sigma_pre is obtained:
  - Mean path:       sigma_pre[i] = sqrt( (W^2)^T var )[i]   — diagonal only
  - Covariance path: cov_pre = W^T cov W,   sigma_pre = sqrt(diag(cov_pre))
"""

from __future__ import annotations

import mechestim as me

from whestbench import BaseEstimator
from whestbench.domain import MLP

# ---------------------------------------------------------------------------
# Helpers: standard normal PDF and CDF
# ---------------------------------------------------------------------------

# Abramowitz & Stegun approximation constants (formula 26.2.17)
_P = 0.2316419
_A1 = 0.319381530
_A2 = -0.356563782
_A3 = 1.781477937
_A4 = -1.821255978
_A5 = 1.330274429


def _norm_pdf(x: me.ndarray) -> me.ndarray:
    """Standard normal PDF: phi(x) = exp(-x^2 / 2) / sqrt(2*pi)."""
    return me.exp(-0.5 * x * x) / me.sqrt(2.0 * me.pi)


def _norm_cdf(x: me.ndarray) -> me.ndarray:
    """Standard normal CDF using the Abramowitz & Stegun approximation.

    Uses only basic mechestim operations (exp, abs). Accurate to < 7.5e-8.
    """
    t = 1.0 / (1.0 + _P * me.abs(x))
    poly = ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    pdf = me.exp(-0.5 * x * x) / me.sqrt(2.0 * me.pi)
    cdf = 1.0 - pdf * poly
    return me.where(x >= 0, cdf, 1.0 - cdf)


# ---------------------------------------------------------------------------
# Mean propagation path  (diagonal variance only)
# ---------------------------------------------------------------------------


def _mean_path(mlp: MLP) -> me.ndarray:
    """Propagate means with a diagonal variance approximation.

    Cost: O(width^2) per layer — scales well to large networks.

    Returns an array of shape (depth, width).
    """
    width = mlp.width

    # Initialise input distribution as standard normal
    mu = me.zeros(width)  # mean vector
    var = me.ones(width)  # per-neuron variance (diagonal of covariance)

    rows = []
    for w in mlp.weights:
        # -- Linear layer --
        # Pre-activation mean:      mu_pre = W^T mu
        mu_pre = w.T @ mu
        # Pre-activation variance:  var_pre[i] = sum_j W[j,i]^2 * var[j]
        var_pre = (w * w).T @ var
        var_pre = me.maximum(var_pre, 1e-12)
        sigma_pre = me.sqrt(var_pre)

        # -- ReLU layer --
        alpha = mu_pre / sigma_pre
        phi_alpha = _norm_pdf(alpha)
        Phi_alpha = _norm_cdf(alpha)

        # E[ReLU(pre)]
        mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

        # Var[ReLU(pre)]
        ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
        var = me.maximum(ez2 - mu * mu, 0.0)

        rows.append(mu)

    return me.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Covariance propagation path  (full covariance matrix)
# ---------------------------------------------------------------------------

# Rescale covariance when any diagonal entry exceeds this value
_COV_RESCALE_THRESHOLD = 1e100


def _covariance_path(mlp: MLP) -> me.ndarray:
    """Propagate means with a full covariance matrix.

    Cost: O(width^3) per layer — more accurate but expensive for wide networks.

    Returns an array of shape (depth, width).
    """
    width = mlp.width

    # Initialise input as standard multivariate normal
    mu = me.zeros(width)  # mean vector
    cov = me.eye(width)  # full covariance matrix
    log_scale = 0.0  # accumulated log of rescaling factor

    rows = []
    for w in mlp.weights:
        # -- Overflow prevention --
        # Rescale (mu, cov) if the covariance has grown too large
        cov_diag = me.diag(cov)
        max_var_np = float(me.max(cov_diag))
        if max_var_np > _COV_RESCALE_THRESHOLD:
            s = float(me.sqrt(max_var_np))
            mu = mu / s
            cov = cov / (s * s)
            log_scale += float(me.log(s))

        # -- Linear layer --
        # Pre-activation mean:         mu_pre  = W^T mu
        # Pre-activation covariance:   cov_pre = W^T cov W
        mu_pre = w.T @ mu
        cov_pre = w.T @ cov @ w

        var_pre = me.maximum(me.diag(cov_pre), 1e-12)
        sigma_pre = me.sqrt(var_pre)

        # -- ReLU layer --
        alpha = mu_pre / sigma_pre
        phi_alpha = _norm_pdf(alpha)
        Phi_alpha = _norm_cdf(alpha)

        # Post-ReLU mean (exact per neuron)
        mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

        # Post-ReLU diagonal variance (exact per neuron)
        ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
        var_post = me.maximum(ez2 - mu * mu, 0.0)

        # Approximate post-ReLU off-diagonal covariance via gain scaling
        sigma_np = me.asarray(sigma_pre, dtype=me.float64)
        Phi_np = me.asarray(Phi_alpha, dtype=me.float64)
        gain_np = me.where(sigma_np > 1e-12, Phi_np, 0.0)
        gain = me.array(gain_np.astype(me.float32))

        cov = me.multiply(me.outer(gain, gain), cov_pre)
        me.fill_diagonal(cov, var_post)  # exact diagonal

        # Record mean in original (unscaled) coordinates
        scale_factor = float(me.exp(log_scale))
        rows.append(mu * scale_factor)

    return me.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Combined (budget-routing) estimator
# ---------------------------------------------------------------------------

# Switch to covariance path when budget allows at least this many FLOPs
# per width^2 (i.e. enough room for the extra matrix operations).
_COVARIANCE_FLOP_MULTIPLIER = 30


class Estimator(BaseEstimator):
    """Budget-aware hybrid estimator.

    Routes to covariance propagation when the FLOP budget is large enough
    relative to width^2, otherwise falls back to (cheaper) mean propagation.

    Decision rule:
        budget >= 30 * width^2  →  _covariance_path(mlp)
        budget <  30 * width^2  →  _mean_path(mlp)
    """

    def predict(self, mlp: MLP, budget: int) -> me.ndarray:
        """Route to the appropriate algorithm based on available FLOP budget.

        Returns an array of shape (depth, width) where row i is the predicted
        mean activation vector after the i-th ReLU layer.
        """
        if budget >= _COVARIANCE_FLOP_MULTIPLIER * mlp.width * mlp.width:
            return _covariance_path(mlp)
        return _mean_path(mlp)

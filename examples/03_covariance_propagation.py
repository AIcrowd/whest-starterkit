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

import whest as we

from whestbench import BaseEstimator
from whestbench.domain import MLP

# If any diagonal entry of the covariance exceeds this value we rescale
# to keep the arithmetic well-behaved in float32.
_COV_RESCALE_THRESHOLD = 1e100


class Estimator(BaseEstimator):
    """Full covariance propagation estimator for ReLU MLPs.

    Tracks the full (width x width) covariance matrix through every layer.
    More accurate than mean propagation for correlated networks, but costs
    O(width^2) memory and O(width^3) FLOPs per layer.
    """

    def predict(self, mlp: MLP, budget: int) -> we.ndarray:
        """Predict per-layer output means via full covariance propagation.

        Returns an array of shape (depth, width) where row i is the predicted
        mean activation vector after the i-th ReLU layer.
        """
        _ = budget  # budget is unused by this estimator
        width = mlp.width

        # --- Step 1: initialise the input distribution ---
        # Input is modelled as standard multivariate normal: mu=0, cov=I.
        mu = we.zeros(width)  # shape (width,)
        cov = we.eye(width)  # shape (width, width)
        log_scale = 0.0  # tracks accumulated log of rescaling factor

        rows = []
        for w in mlp.weights:  # w has shape (width, width)
            # --- Step 2: overflow prevention ---
            # If the covariance has grown very large, rescale (mu, cov) by the
            # square root of the largest variance so that downstream matmuls
            # stay in a safe range.  We compensate in the recorded mean later.
            cov_diag = we.diag(cov)
            max_var_np = float(we.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                s = float(we.sqrt(max_var_np))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(we.log(s))

            # --- Step 3: propagate through the linear layer ---
            # Pre-activation mean:         mu_pre  = W^T mu
            # Pre-activation covariance:   cov_pre = W^T cov W
            mu_pre = w.T @ mu
            cov_pre = w.T @ cov @ w

            # Extract per-neuron pre-activation standard deviations from the
            # diagonal of cov_pre.
            var_pre = we.maximum(we.diag(cov_pre), 1e-12)
            sigma_pre = we.sqrt(var_pre)

            # --- Step 4: compute alpha = mu / sigma for each neuron ---
            alpha = mu_pre / sigma_pre
            phi_alpha = we.stats.norm.pdf(alpha)
            Phi_alpha = we.stats.norm.cdf(alpha)

            # --- Step 5: post-ReLU mean (exact per neuron) ---
            # E[ReLU(pre)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # --- Step 6: post-ReLU diagonal variance (exact per neuron) ---
            # E[z^2] = (mu_pre^2 + var_pre) * Phi(alpha) + mu_pre * sigma_pre * phi(alpha)
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = we.maximum(ez2 - mu * mu, 0.0)

            # --- Step 7: approximate post-ReLU covariance ---
            # gain[i] = Phi(alpha[i])  when sigma_pre[i] > 0, else 0
            sigma_np = we.asarray(sigma_pre, dtype=we.float64)
            Phi_np = we.asarray(Phi_alpha, dtype=we.float64)
            gain_np = we.where(sigma_np > 1e-12, Phi_np, 0.0)
            gain = we.array(gain_np.astype(we.float32))

            # Off-diagonal approximation:  cov_post[i,j] ≈ gain[i]*gain[j]*cov_pre[i,j]
            cov = we.multiply(we.outer(gain, gain), cov_pre)

            # Replace the diagonal with the exact marginal variances.
            we.fill_diagonal(cov, var_post)

            # --- Step 8: record mean in original (unscaled) coordinates ---
            scale_factor = float(we.exp(log_scale))
            rows.append(mu * scale_factor)

        # Stack all layer means into a single (depth, width) array
        return we.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=32, depth=6, seed=0)
    compare_against_monte_carlo(Estimator(), mlp)

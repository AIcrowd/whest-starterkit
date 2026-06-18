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

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

# If any diagonal entry of the covariance exceeds this value we rescale
# to keep the arithmetic well-behaved in float32.
_COV_RESCALE_THRESHOLD = 1e100


class Estimator(BaseEstimator):
    """Full covariance propagation estimator for ReLU MLPs.

    Tracks the full (width x width) covariance matrix through every layer.
    More accurate than mean propagation for correlated networks, but costs
    O(width^2) memory and O(width^3) FLOPs per layer.

    Seeding (whestbench contract -- see
    ``docs/reference/estimator-contract.md``): this estimator is deterministic,
    but it carries the canonical seeding scaffold so the propagation examples
    (01–03) all show the pattern. ``self._setup_rng`` is the submission-level
    RNG seeded from ``ctx.seed`` inside ``setup``; the ``_rng`` line at the top
    of ``predict`` is the per-MLP RNG seeded from ``mlp.seed``. Both are unused
    here because the algorithm is purely analytical.
    """

    def __init__(self) -> None:
        self._setup_rng = None  # set from ctx.seed inside setup()

    def setup(self, ctx: SetupContext) -> None:
        # Submission-level RNG; unused in this deterministic estimator but
        # carried here so every example shows the pattern.
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict per-layer output means via full covariance propagation.

        Returns an array of shape (depth, width) where row i is the predicted
        mean activation vector after the i-th ReLU layer.
        """
        # Per-MLP RNG seeded from the grader's seed; unused here (deterministic
        # algorithm) but carried so every example shows the pattern.
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng  # silences "unused variable" linters
        _ = budget  # budget is unused by this estimator
        width = mlp.width

        # --- Step 1: initialise the input distribution ---
        # Input is modelled as standard multivariate normal: mu=0, cov=I.
        mu = fnp.zeros(width)  # shape (width,)
        cov = fnp.eye(width)  # shape (width, width)
        log_scale = 0.0  # tracks accumulated log of rescaling factor

        rows = []
        for w in mlp.weights:  # w has shape (width, width)
            # --- Step 2: overflow prevention ---
            # If the covariance has grown very large, rescale (mu, cov) by the
            # square root of the largest variance so that downstream matmuls
            # stay in a safe range.  We compensate in the recorded mean later.
            cov_diag = fnp.diag(cov)
            max_var_np = float(fnp.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                s = float(fnp.sqrt(max_var_np))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(fnp.log(s))

            # --- Step 3: propagate through the linear layer ---
            # Pre-activation mean:         mu_pre  = W^T mu
            # Pre-activation covariance:   cov_pre = W^T cov W
            #
            # Use einsum (not the chained matmul `w.T @ cov @ w`) so flopscope
            # detects that the two `w` operands are the same tensor and tags
            # cov_pre as symmetric. Symmetry then flows through the post-ReLU
            # outer-product update below (line ~140), so the resulting `cov`
            # is also tagged symmetric — no SymmetryLossWarning to suppress.
            # See https://github.com/AIcrowd/whestbench/issues/27 for the
            # background.
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

            # Extract per-neuron pre-activation standard deviations from the
            # diagonal of cov_pre.
            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)

            # --- Step 4: compute alpha = mu / sigma for each neuron ---
            alpha = mu_pre / sigma_pre
            phi_alpha = flops.stats.norm.pdf(alpha)
            Phi_alpha = flops.stats.norm.cdf(alpha)

            # --- Step 5: post-ReLU mean (exact per neuron) ---
            # E[ReLU(pre)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # --- Step 6: post-ReLU diagonal variance (exact per neuron) ---
            # E[z^2] = (mu_pre^2 + var_pre) * Phi(alpha) + mu_pre * sigma_pre * phi(alpha)
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)

            # --- Step 7: approximate post-ReLU covariance ---
            # gain[i] = Phi(alpha[i])  when sigma_pre[i] > 0, else 0
            sigma_np = fnp.asarray(sigma_pre, dtype=fnp.float64)
            Phi_np = fnp.asarray(Phi_alpha, dtype=fnp.float64)
            gain_np = fnp.where(sigma_np > 1e-12, Phi_np, 0.0)
            gain = fnp.array(gain_np.astype(fnp.float32))

            # Off-diagonal approximation:  cov_post[i,j] ≈ gain[i]*gain[j]*cov_pre[i,j]
            cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)

            # Replace the diagonal with the exact marginal variances.
            fnp.fill_diagonal(cov, var_post)

            # --- Step 8: record mean in original (unscaled) coordinates ---
            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu * scale_factor)

        # Stack all layer means into a single (depth, width) array
        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)  # phase-1 competition shape (warmup round used depth=8)
    compare_against_monte_carlo(Estimator(), mlp)

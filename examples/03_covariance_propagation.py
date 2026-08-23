"""Covariance propagation estimator for ReLU MLPs, a self-contained educational implementation.

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

    and the estimator replaces the diagonal with the exact marginal variance
    from the ReLU expectation formula:

        var_post[i] = E[z_i^2] - E[z_i]^2

Numerical stability:
    Deep networks can cause the covariance to grow very large.  Before each
    linear layer this estimator checks the maximum diagonal entry and rescales
    (mu, cov) if it exceeds a threshold, keeping a running log-scale to restore
    the mean in the original coordinates before recording it.

    At the graded shape (1024x16, He weights) this guard never fires: the
    largest diagonal entry stays between 0.26 and 1.37 across all 16 layers.
    It is carried for the deeper or differently-scaled networks you may try
    next. Leaving it in costs 1,023 FLOPs per layer, 16,368 across the suite,
    against a 51.7-billion-FLOP predict().
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

# If any diagonal entry of the covariance exceeds this value, Step 2 rescales to
# keep the arithmetic inside the float32 range (max 3.4e38; see the float32
# seeding in Step 1).
# The float32 maximum is 3.4028235e38, so the float64-era 1e100 this constant used
# to hold can never fire: it rounds to inf on conversion, and by the time a
# diagonal entry compared greater the covariance had already overflowed. 1e30 is
# representable and still leaves room for one more W^T cov W before the overflow.
_COV_RESCALE_THRESHOLD = 1e30


class Estimator(BaseEstimator):
    """Full covariance propagation estimator for ReLU MLPs.

    Tracks the full (width x width) covariance matrix through every layer.
    More accurate than mean propagation for correlated networks, but costs
    O(width^2) memory and O(width^3) FLOPs per layer.

    Seeding (whestbench contract -- see
    ``docs/reference/estimator-contract.md``): this estimator is deterministic,
    but it carries the canonical seeding scaffold so examples 01-03 all show
    the pattern. ``self._setup_rng`` is the submission-level
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
        # Tag the covariance as symmetric from the start: as_symmetric()
        # validates the buffer and attaches the tag that earns the symmetric
        # contraction rate in Step 3.  It bills 7n-1 per element-pass with
        # n = width^2.
        #
        # Both arrays are seeded at float32 on purpose. flopscope bills float64 at
        # TWICE the float32 rate, and `fnp.zeros`/`fnp.eye` default to float64.
        # NumPy promotion then carries that through the O(width^3) einsum in Step 3,
        # which is 99.7% of this estimator's cost, whatever dtype the weights arrive
        # as. Measured against float32 weights on the competition shape
        # (width=1024, depth=16): 103,407,495,614 -> 51,709,240,799 FLOPs, a 49.99%
        # saving, with the final-layer MSE unchanged to three significant figures
        # (4.1674e-06 vs 4.1676e-06).
        # Not quite an exact halving: the 32 stats.norm.pdf/cdf calls bill float64 in
        # BOTH runs, so they are common to numerator and denominator.
        # Covariance propagation does not need float64 precision at this width.
        mu = fnp.zeros(width, dtype=fnp.float32)  # shape (width,)
        cov = flops.as_symmetric(
            fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1)
        )  # (width, width)
        log_scale = 0.0  # tracks accumulated log of rescaling factor

        rows = []
        for w in mlp.weights:  # w has shape (width, width)
            # --- Step 2: overflow prevention ---
            # If the covariance has grown very large, rescale (mu, cov) by the
            # square root of the largest variance so that downstream matmuls
            # stay in a safe range.  Step 8 compensates in the recorded mean.
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
            # Use einsum (not the chained matmul `w.T @ cov @ w`) so the whole
            # product is a single contraction over the symmetric-tagged `cov`
            # (see Steps 1 and 7b). flopscope bills it at the symmetric rate,
            # the single biggest saving in this estimator after dtype (~24% per
            # layer; the float32 seeding in Step 1 is worth a further 50%).
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
            # `flops.stats.norm.*` promotes float32 input to float64 to match
            # scipy.stats (flopscope warns about exactly this). Cast straight back,
            # or the promoted result re-infects the loop at the 2x rate and undoes
            # the float32 seeding above.
            phi_alpha = flops.stats.norm.pdf(alpha).astype(fnp.float32)
            Phi_alpha = flops.stats.norm.cdf(alpha).astype(fnp.float32)

            # --- Step 5: post-ReLU mean (exact per neuron) ---
            # E[ReLU(pre)] = mu_pre * Phi(alpha) + sigma_pre * phi(alpha)
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # --- Step 6: post-ReLU diagonal variance (exact per neuron) ---
            # E[z^2] = (mu_pre^2 + var_pre) * Phi(alpha) + mu_pre * sigma_pre * phi(alpha)
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)

            # --- Step 7: approximate post-ReLU covariance ---
            # gain[i] = Phi(alpha[i])  when sigma_pre[i] > 0, else 0
            #
            # The zero is an explicit float32 array, not a bare `0.0`. A Python
            # float literal is a C double, and under flopscope 0.11.0 it promoted
            # this whole `where` to float64: 131,072 FLOPs across the 16 layers
            # instead of 65,536. flopscope 0.12.0 applies NEP 50 weak promotion and
            # bills the float32 rate for the literal too, so the two versions
            # disagreed on this one line. Passing the dtype explicitly bills the
            # float32 rate on both and makes the estimator's total version-stable.
            # `fnp.zeros` is free, so the zero itself bills nothing.
            _zero32 = fnp.zeros((), dtype=fnp.float32)
            gain = fnp.where(sigma_pre > 1e-12, Phi_alpha, _zero32)

            # Off-diagonal approximation:  cov_post[i,j] ≈ gain[i]*gain[j]*cov_pre[i,j]
            # This multiply keeps the symmetry tag: `outer(gain, gain)` is symmetric
            # by construction and `cov_pre` inherited the tag through the einsum, so
            # both operands share a symmetry group and the product stays tagged.
            cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)

            # Replace the diagonal with the exact marginal variances.
            # This write voids the tag: since flopscope 0.10.0 a write into a
            # tagged buffer drops the tag rather than keeping a discount the
            # buffer may no longer earn. flopscope does not warn you here; it
            # stops applying the symmetric discount silently, so Step 7b
            # re-validates rather than waiting for a warning. (A SymmetryLossWarning
            # does exist, but it fires on an op given mismatched symmetry groups,
            # not on this write.)
            fnp.fill_diagonal(cov, var_post)

            # --- Step 7b: re-validate and re-tag the covariance ---
            # as_symmetric() re-checks the written buffer really is symmetric and
            # re-attaches the tag (7,340,031 FLOPs at width 1024, float32), so the
            # next layer's einsum bills at the symmetric rate again: 3,220,700,672
            # instead of 4,292,870,144 for the same contraction untagged — a
            # 1,072,169,472 saving per layer, 24.98%, which repays the re-validation
            # about 146x. Measured by running this estimator with and without the
            # tag on the competition shape: dropping this one line raises einsum
            # cost from 51,531,210,752 to 67,613,752,832 FLOPs (layers 2-16 fall
            # back to the untagged 4,292,870,144 rate), against 124,780,527 FLOPs
            # for the 17 `as_symmetric` calls that keep it.
            # Re-validate-after-write is the supported symmetry idiom under
            # flopscope >= 0.10.0.
            cov = flops.as_symmetric(cov, symmetry=(0, 1))

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

    mlp = build_mlp(width=1024, depth=16, seed=0)  # competition shape
    compare_against_monte_carlo(Estimator(), mlp)

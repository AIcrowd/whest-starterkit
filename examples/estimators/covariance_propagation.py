from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_estimation import BaseEstimator
from network_estimation.domain import MLP


class Estimator(BaseEstimator):
    """Covariance propagation estimator for ReLU MLPs."""

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        _ = budget
        from scipy.stats import norm  # type: ignore[import-untyped]

        width = mlp.width
        mu = np.zeros(width, dtype=np.float64)
        cov = np.eye(width, dtype=np.float64)
        rows = []

        for w in mlp.weights:
            W = w.astype(np.float64)
            mu_pre = W.T @ mu
            cov_pre = W.T @ cov @ W
            var_pre = np.maximum(np.diag(cov_pre), 1e-12)
            sigma_pre = np.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi = norm.pdf(alpha)
            Phi = norm.cdf(alpha)

            mu = mu_pre * Phi + sigma_pre * phi
            ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma_pre * phi
            var_post = np.maximum(ez2 - mu**2, 0.0)
            gain = np.where(sigma_pre > 1e-12, Phi, 0.0)
            cov = np.outer(gain, gain) * cov_pre
            np.fill_diagonal(cov, var_post)
            rows.append(mu.astype(np.float32))

        return np.stack(rows, axis=0)

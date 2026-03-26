from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_estimation import BaseEstimator
from network_estimation.domain import MLP


class Estimator(BaseEstimator):
    """Budget-aware hybrid estimator: routes between mean and covariance propagation."""

    _COVARIANCE_BUDGET_MULTIPLIER = 30

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        if budget >= self._COVARIANCE_BUDGET_MULTIPLIER * mlp.width:
            return self._covariance_path(mlp)
        return self._mean_path(mlp)

    def _mean_path(self, mlp: MLP) -> NDArray[np.float32]:
        from scipy.stats import norm  # type: ignore[import-untyped]

        width = mlp.width
        mu = np.zeros(width, dtype=np.float64)
        var = np.ones(width, dtype=np.float64)
        rows = []
        for w in mlp.weights:
            W = w.astype(np.float64)
            mu_pre = W.T @ mu
            var_pre = np.maximum((W ** 2).T @ var, 1e-12)
            sigma_pre = np.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            mu = mu_pre * norm.cdf(alpha) + sigma_pre * norm.pdf(alpha)
            ez2 = (mu_pre ** 2 + var_pre) * norm.cdf(alpha) + mu_pre * sigma_pre * norm.pdf(alpha)
            var = np.maximum(ez2 - mu ** 2, 0.0)
            rows.append(mu.astype(np.float32))
        return np.stack(rows, axis=0)

    def _covariance_path(self, mlp: MLP) -> NDArray[np.float32]:
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
            ez2 = (mu_pre ** 2 + var_pre) * Phi + mu_pre * sigma_pre * phi
            var_post = np.maximum(ez2 - mu ** 2, 0.0)
            gain = np.where(sigma_pre > 1e-12, Phi, 0.0)
            cov = np.outer(gain, gain) * cov_pre
            np.fill_diagonal(cov, var_post)
            rows.append(mu.astype(np.float32))
        return np.stack(rows, axis=0)

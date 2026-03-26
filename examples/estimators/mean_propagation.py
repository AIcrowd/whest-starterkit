from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_estimation import BaseEstimator
from network_estimation.domain import MLP


class Estimator(BaseEstimator):
    """Mean propagation estimator for ReLU MLPs."""

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        _ = budget
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

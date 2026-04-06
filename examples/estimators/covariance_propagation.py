from __future__ import annotations

import mechestim as me

from network_estimation import BaseEstimator
from network_estimation.domain import MLP
from network_estimation.estimators import CovariancePropagationEstimator as _CovProp


class Estimator(BaseEstimator):
    """Covariance propagation estimator for ReLU MLPs."""

    def __init__(self) -> None:
        self._inner = _CovProp()

    def predict(self, mlp: MLP, budget: int) -> me.ndarray:
        return self._inner.predict(mlp, budget)

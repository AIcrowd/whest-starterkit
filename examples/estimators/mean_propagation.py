from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from circuit_estimation import BaseEstimator
from circuit_estimation.domain import Circuit, Layer


class Estimator(BaseEstimator):
    """Mean propagation tutorial estimator.

    This is the simplest baseline and usually the best estimator to read first.
    We keep one value per wire, its expected value ``E[x]``, and update those
    means layer by layer.

    For one output wire ``i``, the gate equation is:

        y_i = a_i * x_f + b_i * x_s + c_i + p_i * x_f * x_s

    Mean propagation uses:

        E[x_f * x_s] ~= E[x_f] * E[x_s]

    so the update becomes:

        m_i^(l+1) = E[y_i]
                  = a_i * m_f
                    + b_i * m_s
                    + c_i
                    + p_i * m_f * m_s

    where ``m_f = E[x_f]`` and ``m_s = E[x_s]``.

    Intuition: we model average behavior only. If dependencies between wires
    become strong, this approximation can drift because covariance is ignored.
    """

    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        """Yield one prediction row per depth.

        ``budget`` is accepted for interface compatibility but not used here.
        """
        _ = budget
        # Start from unbiased wire means E[x] = 0 at depth 0.
        x_mean: NDArray[np.float32] = np.zeros(circuit.n, dtype=np.float32)
        for layer in circuit.gates:
            # Push means through one layer of affine+bivariate gates.
            x_mean = self._propagate_layer_mean(layer, x_mean)
            yield x_mean

    @staticmethod
    def _propagate_layer_mean(layer: Layer, x_mean: NDArray[np.float32]) -> NDArray[np.float32]:
        """Propagate one layer of means under mean propagation closure."""
        first_mean = np.take(x_mean, layer.first)
        second_mean = np.take(x_mean, layer.second)
        return (
            layer.first_coeff * first_mean
            + layer.second_coeff * second_mean
            + layer.const
            + layer.product_coeff * first_mean * second_mean
        ).astype(np.float32)

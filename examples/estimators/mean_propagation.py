from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from circuit_estimation import BaseEstimator
from circuit_estimation.domain import Circuit, Layer


class Estimator(BaseEstimator):
    """Mean propagation starter estimator.

    Tracks only wire means E[x] and applies:
        E[x_f * x_s] ~= E[x_f] * E[x_s]

    This is a fast baseline and a good starting point for participants.
    """

    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        _ = budget
        x_mean: NDArray[np.float32] = np.zeros(circuit.n, dtype=np.float32)
        for layer in circuit.gates:
            x_mean = self._propagate_layer_mean(layer, x_mean)
            yield x_mean

    @staticmethod
    def _propagate_layer_mean(layer: Layer, x_mean: NDArray[np.float32]) -> NDArray[np.float32]:
        first_mean = np.take(x_mean, layer.first)
        second_mean = np.take(x_mean, layer.second)
        return (
            layer.first_coeff * first_mean
            + layer.second_coeff * second_mean
            + layer.const
            + layer.product_coeff * first_mean * second_mean
        ).astype(np.float32)

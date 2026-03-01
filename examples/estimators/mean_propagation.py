from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from circuit_estimation import BaseEstimator
from circuit_estimation.domain import Circuit


class Estimator(BaseEstimator):
    """Starter estimator that uses first-moment propagation only."""

    def predict(self, circuit: object, budget: int) -> NDArray[np.float32]:
        typed_circuit = cast(Circuit, circuit)
        x_mean: NDArray[np.float32] = np.zeros(typed_circuit.n, dtype=np.float32)
        outputs = np.zeros((typed_circuit.d, typed_circuit.n), dtype=np.float32)
        for i, layer in enumerate(typed_circuit.gates):
            first_mean = np.take(x_mean, layer.first)
            second_mean = np.take(x_mean, layer.second)
            x_mean = (
                layer.first_coeff * first_mean
                + layer.second_coeff * second_mean
                + layer.const
                + layer.product_coeff * first_mean * second_mean
            ).astype(np.float32)
            outputs[i] = x_mean
        return outputs

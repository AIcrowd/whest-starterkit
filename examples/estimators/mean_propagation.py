from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from circuit_estimation import BaseEstimator
from circuit_estimation.domain import Circuit, Layer


class Estimator(BaseEstimator):
    """First-moment propagation tutorial estimator.

    Overview:
        This estimator tracks only the mean of each wire after each layer.
        It is the simplest useful baseline and demonstrates the minimum
        `Estimator.predict(...)` contract.

    Math:
        Let x_j^(l) in {-1, +1} be wire j at layer l, and one gate output be:

            y_i = a_i * x_f + b_i * x_s + c_i + p_i * x_f * x_s

        We propagate first moments with the closure:

            m_i^(l+1) = E[y_i]
                      = a_i * E[x_f]
                        + b_i * E[x_s]
                        + c_i
                        + p_i * E[x_f] * E[x_s]

        This last term uses the first-moment independence approximation:
        E[x_f * x_s] ~= E[x_f] * E[x_s].

    ASCII:
        layer l means     gate map              layer l+1 means
        m(f), m(s)  ----> y = a*f+b*s+c+p*f*s ----> m'

        Repeat this update for every layer and stack the m' vectors.

    Complexity:
        Time:  O(depth * width)
        Space: O(width) for running state, O(depth * width) for output tensor.

    Pitfall:
        Because this model ignores covariance, it can underperform when
        product terms dominate and strong wire dependencies emerge.
        The `budget` argument is accepted for API compatibility but not used.
    """

    def predict(self, circuit: object, budget: int) -> NDArray[np.float32]:
        typed_circuit = cast(Circuit, circuit)
        _ = budget
        x_mean: NDArray[np.float32] = np.zeros(typed_circuit.n, dtype=np.float32)
        outputs = np.zeros((typed_circuit.d, typed_circuit.n), dtype=np.float32)
        for i, layer in enumerate(typed_circuit.gates):
            x_mean = self._propagate_layer_mean(layer, x_mean)
            outputs[i] = x_mean
        return outputs

    @staticmethod
    def _propagate_layer_mean(layer: Layer, x_mean: NDArray[np.float32]) -> NDArray[np.float32]:
        """Propagate one layer of means under first-moment closure."""
        first_mean = np.take(x_mean, layer.first)
        second_mean = np.take(x_mean, layer.second)
        return (
            layer.first_coeff * first_mean
            + layer.second_coeff * second_mean
            + layer.const
            + layer.product_coeff * first_mean * second_mean
        ).astype(np.float32)

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from circuit_estimation import BaseEstimator
from circuit_estimation.domain import Circuit, Layer


class Estimator(BaseEstimator):
    """Mean propagation estimator.

    This is the simplest baseline and usually the best place to start when
    learning the estimator API. It tracks only the mean of each wire after
    each layer and demonstrates the minimum `Estimator.predict(...)` contract.

    Let x_j^(l) in {-1, +1} be wire j at layer l and define one gate output:

        y_i = a_i * x_f + b_i * x_s + c_i + p_i * x_f * x_s

    The propagated mean is:

        m_i^(l+1) = E[y_i]
                  = a_i * E[x_f]
                    + b_i * E[x_s]
                    + c_i
                    + p_i * E[x_f] * E[x_s]

    where we use the mean-propagation closure:

        E[x_f * x_s] ~= E[x_f] * E[x_s]

    A quick mental model is: layer means -> gate map -> next-layer means, then
    repeat and stack each layer's mean vector into a `(depth, width)` output.

    Runtime is O(depth * width) with O(width) rolling state (plus output storage).
    Because covariance is ignored, this method can underperform when product
    terms induce strong dependencies. The `budget` argument is accepted for API
    compatibility but not used.
    """

    def predict(self, circuit: Circuit, budget: int) -> NDArray[np.float32]:
        _ = budget
        # Start from unbiased wire means E[x] = 0 at depth 0.
        x_mean: NDArray[np.float32] = np.zeros(circuit.n, dtype=np.float32)
        outputs = np.zeros((circuit.d, circuit.n), dtype=np.float32)
        for i, layer in enumerate(circuit.gates):
            # Push means through this layer's affine+bivariate gate map.
            x_mean = self._propagate_layer_mean(layer, x_mean)
            outputs[i] = x_mean
        return outputs

    @staticmethod
    def _propagate_layer_mean(layer: Layer, x_mean: NDArray[np.float32]) -> NDArray[np.float32]:
        """Propagate one layer of means under mean-propagation closure."""
        first_mean = np.take(x_mean, layer.first)
        second_mean = np.take(x_mean, layer.second)
        return (
            layer.first_coeff * first_mean
            + layer.second_coeff * second_mean
            + layer.const
            + layer.product_coeff * first_mean * second_mean
        ).astype(np.float32)

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from circuit_estimation import BaseEstimator
from circuit_estimation.domain import Circuit, Layer


class Estimator(BaseEstimator):
    """Budget-aware hybrid tutorial estimator.

    Overview:
        This class demonstrates a practical pattern used in competitions:
        choose a cheaper estimator at low budget and a more accurate estimator
        at high budget. Here we switch between:
        - first-moment mean propagation
        - pairwise moment closure with covariance tracking

        The switch rule is:
            use covariance when budget >= 30 * circuit.n
            else use mean propagation

    Math:
        Mean branch:
            m_i^(l+1) = a_i * m_f + b_i * m_s + c_i + p_i * m_f * m_s

        Covariance branch:
            tracks both:
                m = E[x]
                C = Cov[x]
            and adds second-order closure terms:
                E[x_f x_s] ~= m_f * m_s + C_fs

        The covariance branch decomposes new covariance into:
            1) linear-linear
            2) 1v2 cross terms
            3) 2v2 bilinear terms

    ASCII:
        Input: (circuit, budget)
                   |
                   v
             +-----------------+
             | budget threshold|
             | 30 * width      |
             +-----------------+
               /            \\
              /              \\
     low budget                high budget
      (fast path)               (accurate path)
         |                           |
         v                           v
       mean                     covariance
    propagation                 propagation
         \\                           /
          \\                         /
           +-----------------------+
           | predictions (d x n)   |
           +-----------------------+

    Complexity:
        Mean branch:       O(depth * n)
        Covariance branch: O(depth * n^2)
        This estimator acts as a policy over those two cost profiles.

    Pitfall:
        A hard threshold can be suboptimal around the decision boundary.
        In production, consider calibrating threshold(s) with offline sweeps.
        This implementation keeps a fixed simple rule for clarity.
    """

    _COVARIANCE_BUDGET_MULTIPLIER = 30

    def predict(self, circuit: object, budget: int) -> NDArray[np.float32]:
        typed_circuit = cast(Circuit, circuit)
        if self._should_use_covariance(typed_circuit.n, budget):
            return self._covariance_propagation(typed_circuit)
        return self._mean_propagation(typed_circuit)

    @classmethod
    def _should_use_covariance(cls, width: int, budget: int) -> bool:
        """Return whether this input should use covariance propagation."""
        return budget >= cls._COVARIANCE_BUDGET_MULTIPLIER * width

    def _mean_propagation(self, circuit: Circuit) -> NDArray[np.float32]:
        """Fast first-moment path used for low budgets."""
        x_mean: NDArray[np.float32] = np.zeros(circuit.n, dtype=np.float32)
        outputs = np.zeros((circuit.d, circuit.n), dtype=np.float32)
        for i, layer in enumerate(circuit.gates):
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

    def _covariance_propagation(self, circuit: Circuit) -> NDArray[np.float32]:
        """Accurate second-order path used for sufficiently high budgets."""
        n = circuit.n
        x_mean: NDArray[np.float32] = np.zeros(n, dtype=np.float32)
        x_cov: NDArray[np.float32] = np.eye(n, dtype=np.float32)
        outputs = np.zeros((circuit.d, n), dtype=np.float32)

        for i, layer in enumerate(circuit.gates):
            x_mean, x_cov = self._propagate_layer_covariance(layer, x_mean, x_cov)
            outputs[i] = x_mean

        return outputs

    def _propagate_layer_covariance(
        self,
        layer: Layer,
        x_mean: NDArray[np.float32],
        x_cov: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return `(new_mean, new_cov)` for one layer under pairwise closure."""
        n = x_mean.shape[0]
        first_mean = x_mean[layer.first]
        second_mean = x_mean[layer.second]
        pair_cov = x_cov[layer.first, layer.second]

        new_mean: NDArray[np.float32] = (
            layer.first_coeff * first_mean
            + layer.second_coeff * second_mean
            + layer.const
            + layer.product_coeff * first_mean * second_mean
            + layer.product_coeff * pair_cov
        ).astype(np.float32)

        new_cov: NDArray[np.float32] = self._linear_linear_covariance(layer, x_cov, n)

        result_1v2_first = np.outer(
            layer.first_coeff,
            layer.product_coeff,
        ) * self._one_v_two_covariance(layer.first, layer.first, layer.second, x_cov, x_mean)
        new_cov += result_1v2_first + result_1v2_first.T

        result_1v2_second = np.outer(
            layer.second_coeff,
            layer.product_coeff,
        ) * self._one_v_two_covariance(layer.second, layer.first, layer.second, x_cov, x_mean)
        new_cov += result_1v2_second + result_1v2_second.T

        new_cov += np.outer(layer.product_coeff, layer.product_coeff) * self._two_v_two_covariance(
            layer.first, layer.second, layer.first, layer.second, x_cov, x_mean
        )

        self._clip_moments(new_mean, new_cov)
        return new_mean, new_cov

    @staticmethod
    def _linear_linear_covariance(
        layer: Layer,
        x_cov: NDArray[np.float32],
        n: int,
    ) -> NDArray[np.float32]:
        """Compute covariance contribution from linear terms only."""
        new_cov = np.zeros((n, n), dtype=np.float32)
        new_cov += (
            np.outer(layer.first_coeff, layer.first_coeff) * x_cov[np.ix_(layer.first, layer.first)]
        )
        new_cov += (
            np.outer(layer.second_coeff, layer.second_coeff)
            * x_cov[np.ix_(layer.second, layer.second)]
        )
        new_cov += (
            np.outer(layer.first_coeff, layer.second_coeff)
            * x_cov[np.ix_(layer.first, layer.second)]
        )
        new_cov += (
            np.outer(layer.second_coeff, layer.first_coeff)
            * x_cov[np.ix_(layer.second, layer.first)]
        )
        return new_cov

    @staticmethod
    def _one_v_two_covariance(
        a: NDArray[np.int32],
        b: NDArray[np.int32],
        c: NDArray[np.int32],
        x_cov: NDArray[np.float32],
        x_mean: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Approximate Cov[x_a, x_b * x_c] under pairwise closure."""
        return (
            x_mean[b][None, :] * x_cov[np.ix_(a, c)] + x_mean[c][None, :] * x_cov[np.ix_(a, b)]
        ).astype(np.float32)

    @staticmethod
    def _two_v_two_covariance(
        a: NDArray[np.int32],
        b: NDArray[np.int32],
        c: NDArray[np.int32],
        d: NDArray[np.int32],
        cov: NDArray[np.float32],
        mean: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Approximate Cov[x_a * x_b, x_c * x_d] under pairwise closure."""
        cov_ac = cov[np.ix_(a, c)]
        cov_ad = cov[np.ix_(a, d)]
        cov_bc = cov[np.ix_(b, c)]
        cov_bd = cov[np.ix_(b, d)]

        mu_a = mean[a][:, None]
        mu_b = mean[b][:, None]
        mu_c = mean[c][None, :]
        mu_d = mean[d][None, :]

        return (
            (mu_a * mu_c) * cov_bd
            + (mu_a * mu_d) * cov_bc
            + (mu_b * mu_c) * cov_ad
            + (mu_b * mu_d) * cov_ac
        ).astype(np.float32)

    @staticmethod
    def _clip_moments(mean: NDArray[np.float32], cov: NDArray[np.float32]) -> None:
        """Project moments to a numerically stable feasible region."""
        n = len(mean)
        np.clip(mean, -1.0, 1.0, out=mean)
        var = 1.0 - mean * mean
        cov[np.arange(n), np.arange(n)] = var
        std = np.sqrt(np.clip(var, 0.0, None))
        max_cov = np.outer(std, std)
        np.clip(cov, -max_cov, max_cov, out=cov)

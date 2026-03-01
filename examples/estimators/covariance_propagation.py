from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from circuit_estimation import BaseEstimator
from circuit_estimation.domain import Circuit, Layer


class Estimator(BaseEstimator):
    """Pairwise moment closure tutorial estimator (mean + covariance).

    This is the "next step" after mean propagation when you want to model
    interactions between wires. It extends the state from just means to both
    means and covariance, which improves accuracy when product terms dominate.

    At each depth we track m = E[x] (shape n) and C = Cov[x] (shape n x n).
    For y_i = a_i * x_f + b_i * x_s + c_i + p_i * x_f * x_s, the mean update is
    E[y_i] = a_i * m_f + b_i * m_s + c_i + p_i * (m_f * m_s + C_fs). Covariance
    is updated via pairwise closure by combining linear-linear, linear-bilinear,
    and bilinear-bilinear contributions.

    A useful mental model is a layered state machine:
    (m, C) at depth l -> decomposed covariance update blocks -> (m', C') at l+1.
    The helper methods map directly to those blocks so the code is easy to read.

    Runtime is O(depth * n^2) with O(n^2) rolling state (plus output storage).
    Pairwise closure is still approximate, so strongly non-Gaussian regimes can
    leak higher moments; `_clip_moments` keeps values in a numerically stable,
    feasible region after each layer.
    """

    def predict(self, circuit: object, budget: int) -> NDArray[np.float32]:
        typed_circuit = cast(Circuit, circuit)
        _ = budget
        n = typed_circuit.n
        # Initialize the moment state for random {-1,+1} inputs.
        x_mean: NDArray[np.float32] = np.zeros(n, dtype=np.float32)
        x_cov: NDArray[np.float32] = np.eye(n, dtype=np.float32)
        outputs = np.zeros((typed_circuit.d, n), dtype=np.float32)

        for i, layer in enumerate(typed_circuit.gates):
            x_mean, x_cov = self._propagate_layer(layer, x_mean, x_cov)
            outputs[i] = x_mean

        return outputs

    def _propagate_layer(
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

        # Step 1: propagate means with covariance-corrected product expectation.
        new_mean: NDArray[np.float32] = (
            layer.first_coeff * first_mean
            + layer.second_coeff * second_mean
            + layer.const
            + layer.product_coeff * first_mean * second_mean
            + layer.product_coeff * pair_cov
        ).astype(np.float32)

        # Step 2: start covariance with linear-linear contributions.
        new_cov: NDArray[np.float32] = self._linear_linear_covariance(layer, x_cov, n)

        # Step 3: add linear-bilinear cross contributions (1v2 terms).
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

        # Step 4: add bilinear-bilinear contribution (2v2 term).
        new_cov += np.outer(layer.product_coeff, layer.product_coeff) * self._two_v_two_covariance(
            layer.first, layer.second, layer.first, layer.second, x_cov, x_mean
        )

        # Step 5: clip moments back into a feasible/stable region.
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

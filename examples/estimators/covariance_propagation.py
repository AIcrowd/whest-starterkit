from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from circuit_estimation import BaseEstimator
from circuit_estimation.domain import Circuit


def _one_v_two_covariance(
    a: NDArray[np.int32],
    b: NDArray[np.int32],
    c: NDArray[np.int32],
    x_cov: NDArray[np.float32],
    x_mean: NDArray[np.float32],
) -> NDArray[np.float32]:
    return (
        x_mean[b][None, :] * x_cov[np.ix_(a, c)] + x_mean[c][None, :] * x_cov[np.ix_(a, b)]
    ).astype(np.float32)


def _two_v_two_covariance(
    a: NDArray[np.int32],
    b: NDArray[np.int32],
    c: NDArray[np.int32],
    d: NDArray[np.int32],
    cov: NDArray[np.float32],
    mean: NDArray[np.float32],
) -> NDArray[np.float32]:
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


def _clip(mean: NDArray[np.float32], cov: NDArray[np.float32]) -> None:
    n = len(mean)
    np.clip(mean, -1.0, 1.0, out=mean)
    var = 1.0 - mean * mean
    cov[np.arange(n), np.arange(n)] = var
    std = np.sqrt(np.clip(var, 0.0, None))
    max_cov = np.outer(std, std)
    np.clip(cov, -max_cov, max_cov, out=cov)


class Estimator(BaseEstimator):
    """Starter estimator with pairwise moment closure for covariance tracking."""

    def predict(self, circuit: object, budget: int) -> NDArray[np.float32]:
        typed_circuit = cast(Circuit, circuit)
        n = typed_circuit.n
        x_mean: NDArray[np.float32] = np.zeros(n, dtype=np.float32)
        x_cov: NDArray[np.float32] = np.eye(n, dtype=np.float32)
        outputs = np.zeros((typed_circuit.d, n), dtype=np.float32)

        for i, layer in enumerate(typed_circuit.gates):
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

            new_cov: NDArray[np.float32] = np.zeros((n, n), dtype=np.float32)
            new_cov += (
                np.outer(layer.first_coeff, layer.first_coeff)
                * x_cov[np.ix_(layer.first, layer.first)]
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

            result_1v2_first = np.outer(
                layer.first_coeff,
                layer.product_coeff,
            ) * _one_v_two_covariance(layer.first, layer.first, layer.second, x_cov, x_mean)
            new_cov += result_1v2_first + result_1v2_first.T

            result_1v2_second = np.outer(
                layer.second_coeff,
                layer.product_coeff,
            ) * _one_v_two_covariance(layer.second, layer.first, layer.second, x_cov, x_mean)
            new_cov += result_1v2_second + result_1v2_second.T

            new_cov += np.outer(
                layer.product_coeff,
                layer.product_coeff,
            ) * _two_v_two_covariance(layer.first, layer.second, layer.first, layer.second, x_cov, x_mean)

            _clip(new_mean, new_cov)
            x_mean, x_cov = new_mean, new_cov
            outputs[i] = x_mean

        return outputs

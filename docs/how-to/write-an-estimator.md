# Write an Estimator

## When To Use This Page

Use this page when implementing your custom participant estimator.

## Minimal Structure

Your estimator should subclass `BaseEstimator` and implement `predict`.

```python
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from circuit_estimation import BaseEstimator
from circuit_estimation.domain import Circuit


class Estimator(BaseEstimator):
    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        x_mean = np.zeros(circuit.n, dtype=np.float32)
        for _layer in circuit.gates:
            yield x_mean
```

## Contract Checklist

- emit exactly `circuit.d` rows,
- each row must be shape `(circuit.n,)`,
- rows must contain finite values,
- stream with `yield` (do not return one final tensor).

## Recommended Learning Path

1. `examples/estimators/random_estimator.py`
2. `examples/estimators/mean_propagation.py`
3. `examples/estimators/covariance_propagation.py`
4. `examples/estimators/combined_estimator.py`

## Next

- [Estimator Contract](../reference/estimator-contract.md)
- [Validate, Run, and Package](./validate-run-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)

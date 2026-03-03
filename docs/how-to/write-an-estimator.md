# Write an Estimator

## When to use this page

Use this page when implementing your custom participant estimator.

## Do this now

Start from [`examples/estimators/random_estimator.py`](../../examples/estimators/random_estimator.py), then replace the prediction logic.

Minimal structure:

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

## ✅ Expected outcome

Your estimator implements `predict(circuit, budget)` and yields one valid depth row at a time.

## Circuit traversal starter

If you need exact `Circuit` / `Layer` field semantics or packed tensors from `circuit.to_vectorized()`, use:

- [Inspect and Traverse Circuit Structure](./inspect-circuit-structure.md)

## 📌 Contract checklist

- emit exactly `circuit.d` rows,
- each row must be shape `(circuit.n,)`,
- all values must be finite,
- stream with `yield` (do not return one final tensor).

## Recommended learning path

1. [`examples/estimators/random_estimator.py`](../../examples/estimators/random_estimator.py)
2. [`examples/estimators/mean_propagation.py`](../../examples/estimators/mean_propagation.py)
3. [`examples/estimators/covariance_propagation.py`](../../examples/estimators/covariance_propagation.py)
4. [`examples/estimators/combined_estimator.py`](../../examples/estimators/combined_estimator.py)

## 🛠 Common first failure

Symptom: estimator returns a full `(depth, width)` array at the end.

Fix: make `predict` a generator and `yield` one `(width,)` row per layer.

## ➡️ Next step

- [Inspect and Traverse Circuit Structure](./inspect-circuit-structure.md)
- [Estimator Contract](../reference/estimator-contract.md)
- [Validate, Run, and Package](./validate-run-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)

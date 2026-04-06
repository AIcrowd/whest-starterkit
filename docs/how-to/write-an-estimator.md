# Write an Estimator

## When to use this page

Use this page when implementing your custom participant estimator.

## Do this now

Start from [`examples/estimators/random_estimator.py`](../../examples/estimators/random_estimator.py), then replace the prediction logic.

Minimal structure:

```python
from __future__ import annotations

import mechestim as me

from network_estimation import BaseEstimator
from network_estimation.domain import MLP


class Estimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> me.ndarray:
        return me.zeros((mlp.depth, mlp.width))
```

## Expected outcome

Your estimator implements `predict(mlp, budget)` and returns a `(depth, width)` array of predicted neuron values.

## MLP traversal starter

If you need exact `MLP` field semantics or weight matrices, use:

- [Inspect and Traverse MLP Structure](./inspect-mlp-structure.md)

## Contract checklist

- return a `(mlp.depth, mlp.width)` array,
- all values must be finite.

## Common first failure

Symptom: estimator returns wrong shape.

Fix: ensure `predict` returns a 2D array with shape `(mlp.depth, mlp.width)` and all finite values.

---

## Building your first estimator

### Step 1: The zeros baseline

The template estimator returns all zeros. Run it to see what a bad score looks like:

```bash
nestim run --estimator ./my-estimator/estimator.py --n-mlps 3
```

Look at `primary_score` — this is the MSE of predicting all zeros. It is your floor.

### Step 2: Mean propagation

Copy the mean propagation example — it uses the ReLU expectation formula:

```bash
cp examples/estimators/mean_propagation.py ./my-estimator/estimator.py
nestim run --estimator ./my-estimator/estimator.py --n-mlps 3
```

Compare `primary_score` to the zeros baseline. Mean propagation uses the network's weights to make informed predictions, so it should score significantly better.

### Step 3: Understand the score report

The report shows per-MLP results:
- `final_mse`: your accuracy on the final layer (primary ranking metric)
- `flops_used`: how many FLOPs your estimator consumed
- `budget_exhausted`: whether you exceeded the budget (predictions zeroed if true)

### Step 4: Try the combined estimator

The combined estimator routes between cheap and expensive algorithms based on budget:

```bash
cp examples/estimators/combined_estimator.py ./my-estimator/estimator.py
nestim run --estimator ./my-estimator/estimator.py --n-mlps 3
```

This demonstrates the budget-aware routing pattern — a common design for production estimators.

---

## Recommended learning path

1. [`examples/estimators/random_estimator.py`](../../examples/estimators/random_estimator.py) — the interface
2. [`examples/estimators/mean_propagation.py`](../../examples/estimators/mean_propagation.py) — simplest real algorithm
3. [`examples/estimators/covariance_propagation.py`](../../examples/estimators/covariance_propagation.py) — more accurate, more expensive
4. [`examples/estimators/combined_estimator.py`](../../examples/estimators/combined_estimator.py) — budget-aware routing
5. [Algorithm Ideas](./algorithm-ideas.md) — full survey of strategies
6. [Performance Tips](./performance-tips.md) — FLOP optimization patterns

## Next step

- [Inspect and Traverse MLP Structure](./inspect-mlp-structure.md)
- [Algorithm Ideas](./algorithm-ideas.md)
- [Manage FLOP Budget](./manage-flop-budget.md)
- [Estimator Contract](../reference/estimator-contract.md)
- [Validate, Run, and Package](./validate-run-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)

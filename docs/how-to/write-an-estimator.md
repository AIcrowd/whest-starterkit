# Write an Estimator

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page when implementing your custom participant estimator.

## 🚀 Do this now

Start from [`examples/01_random.py`](../../examples/01_random.py), then replace the prediction logic.

Minimal structure:

```python
from __future__ import annotations

import flopscope.numpy as fnp

from whestbench import BaseEstimator, MLP


class Estimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return fnp.zeros((mlp.depth, mlp.width))
```

## ✅ Expected outcome

Your estimator implements `predict(mlp, budget)` and returns a `(depth, width)` array of predicted neuron values. At the Phase 2 competition shape — width 1024, depth 16 — that is `(16, 1024)`, and `budget` is `2,199,023,255,552` FLOPs (2^41) for the MLP in front of you.

## MLP traversal starter

If you need exact `MLP` field semantics or weight matrices, use:

- [Inspect and Traverse MLP Structure](./inspect-mlp-structure.md)

## Contract checklist

- return a `(mlp.depth, mlp.width)` array,
- all values must be finite,
- read the shape off `mlp` — never hard-code it.

## ⚠️ Common first failure

Symptom: estimator returns wrong shape.

Fix: ensure `predict` returns a 2D array with shape `(mlp.depth, mlp.width)` and all finite values. A `(32, 256)` constant carried over from Phase 1 is the current form of this bug — the graded shape is now `(16, 1024)`, and `mlp.depth` / `mlp.width` hand it to you with no constant to forget.

---

## Building your first estimator

### Step 1: The zeros baseline

The template estimator returns all zeros. Run it to see what a bad score looks like:

```bash
uv run whest run --estimator estimator.py --n-mlps 3
```

Look at `final_layer_mse` — the raw accuracy of predicting all zeros — and `adjusted_final_layer_score`, the budget-scaled metric the leaderboard ranks on. The zeros baseline is your accuracy floor.

### Step 2: Mean propagation

Copy the mean propagation example — it uses the ReLU expectation formula:

```bash
cp examples/02_mean_propagation.py estimator.py
uv run whest run --estimator estimator.py --n-mlps 3
```

Compare `adjusted_final_layer_score` (and the underlying `final_layer_mse`) to the zeros baseline. Mean propagation uses the network's weights to make informed predictions, so it should score significantly better.

It is also cheap: at the competition shape it spends **86,639,616 FLOPs** per `predict()` call, 0.004% of the 2^41 per-MLP budget. That leaves the compute multiplier at its `0.1` floor, so at this price point your ranking moves only when your accuracy does.

### Step 3: Understand the score report

The report shows per-MLP results:
- `final_layer_mse`: your raw accuracy on the final layer (the diagnostic that drives your score)
- `adjusted_final_layer_score`: `final_layer_mse` scaled by your compute use — the leaderboard ranking metric
- `flops_used`: how many FLOPs your estimator consumed
- `budget_exhausted`: whether you exceeded the budget (predictions zeroed if true)

## Recommended learning path

1. [`examples/01_random.py`](../../examples/01_random.py) — the interface
2. [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py) — simplest real algorithm
3. [`examples/03_covariance_propagation.py`](../../examples/03_covariance_propagation.py) — more accurate, and far more expensive: **51,709,240,799 FLOPs** per `predict()` at the competition shape, 2.351% of the per-MLP budget and 597× the cost of mean propagation. That is still under the 10% of budget below which the score multiplier sits on its `0.1` floor — but the width-cubed `einsum` is 99.67% of the bill, so the margin narrows fast if you add anything on top
4. [`estimator.py`](../../estimator.py) — the repo-root template, runnable two ways: `uv run python estimator.py` for the pure-local pedagogical loop (see [Stage 1](../getting-started/stage-1-standalone.md)) and `uv run whest run --estimator estimator.py` for the harness path. Copy when you want a minimal iteration loop.
5. [Algorithm Ideas](./algorithm-ideas.md) — full survey of strategies
6. [Performance Tips](./performance-tips.md) — FLOP optimization patterns

## ➡️ Next step

- [Inspect and Traverse MLP Structure](./inspect-mlp-structure.md)
- [Algorithm Ideas](./algorithm-ideas.md)
- [Manage FLOP Budget](./manage-flop-budget.md)
- [Estimator Contract](../reference/estimator-contract.md)
- [Validate, Run, and Package](./validate-run-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)

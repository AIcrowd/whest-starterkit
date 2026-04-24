# Stage 2: Validate the Contract

Stage 1 confirms your estimator runs and converges. Stage 2 confirms it satisfies the harness contract — right shapes, right types, no exceptions on edge cases.

## Run it

```bash
whest validate --estimator estimator.py
```

If your estimator is contract-compliant, you'll see a green check. If not, the validator prints the specific contract violation (wrong shape, missing method, etc.).

## What `whest validate` actually checks

- `Estimator` class is importable from `estimator.py`
- `predict(mlp, budget)` exists and returns `whest.ndarray`
- Returned shape is `(mlp.depth, mlp.width)`
- No exceptions on edge MLPs (width=1, depth=1)

See [docs/reference/estimator-contract.md](../reference/estimator-contract.md) for the full contract.

## When you're ready

Move on to [Stage 3: run the harness locally](stage-3-run-local.md).

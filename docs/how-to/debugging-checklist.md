# Debugging Checklist

Use this page when your estimator runs but the score is bad, or something feels wrong. Work through the tiers in order.

## Tier 1: Sanity checks (2 minutes)

Run validation:

```bash
whest validate --estimator ./my-estimator/estimator.py
```

If it fails, check:

- [ ] **Output shape:** does `predict()` return shape `(mlp.depth, mlp.width)`?
- [ ] **Finite values:** are all values finite? Check for `nan` or `inf` in your math.
- [ ] **Class name:** is your class named `Estimator`? The loader looks for this by default.

## Tier 2: Correctness checks (5 minutes)

Run your estimator and look at the report:

```bash
whest run --estimator ./my-estimator/estimator.py --n-mlps 3 --runner local --debug
```

Check:

- [ ] **Does zeros beat you?** If returning `me.zeros((mlp.depth, mlp.width))` scores better than your estimator, your predictions are wrong in a way that's worse than guessing zero.
- [ ] **Is `budget_exhausted` true?** If so, your estimator exceeded the FLOP budget and all predictions were zeroed. See [Manage Your FLOP Budget](./manage-flop-budget.md).
- [ ] **Are errors concentrated at deep layers?** Run with `--debug` and compare `all_layer_mse` — if early layers are good but later layers are bad, your propagation may accumulate errors.

## Tier 3: Optimization checks (10+ minutes)

Profile your FLOP usage:

```python
import mechestim as me

with me.BudgetContext(flop_budget=100_000_000) as budget:
    result = estimator.predict(mlp, budget=100_000_000)
    me.budget_summary()
```

Check:

- [ ] **Is matmul dominant?** If >90% of FLOPs are in matmul, consider diagonal variance instead of full covariance.
- [ ] **Redundant computation?** Are you computing something in a loop that could be precomputed once?
- [ ] **Free operations wasted?** Remember: `me.zeros`, `me.transpose`, `me.reshape`, indexing cost 0 FLOPs.

## Next step

- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
- [Performance Tips](./performance-tips.md)
- [Manage Your FLOP Budget](./manage-flop-budget.md)

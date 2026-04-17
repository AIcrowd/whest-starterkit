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

- [ ] **Did `predict()` raise?** If `whest run` exits with status `1` and prints an "Estimator Errors" panel, your estimator raised an exception. Use `--debug` to include tracebacks inline in the panel, or add `--fail-fast` to halt at the first failure and let the raw Python traceback propagate.
- [ ] **Does zeros beat you?** If returning `we.zeros((mlp.depth, mlp.width))` scores better than your estimator, your predictions are wrong in a way that's worse than guessing zero.
- [ ] **Is `budget_exhausted` true?** If so, your estimator exceeded the FLOP budget and all predictions were zeroed. See [Manage Your FLOP Budget](./manage-flop-budget.md).
- [ ] **Are errors concentrated at deep layers?** Run with `--debug` and compare `all_layer_mse` — if early layers are good but later layers are bad, your propagation may accumulate errors.

## Tier 3: Optimization checks (10+ minutes)

Profile your FLOP usage:

```python
import whest as we

with we.BudgetContext(flop_budget=100_000_000) as budget:
    result = estimator.predict(mlp, budget=100_000_000)
    we.budget_summary()
```

Check:

- [ ] **Is matmul dominant?** If >90% of FLOPs are in matmul, consider diagonal variance instead of full covariance.
- [ ] **Redundant computation?** Are you computing something in a loop that could be precomputed once?
- [ ] **Free operations wasted?** Remember: `we.zeros`, `we.transpose`, `we.reshape`, indexing cost 0 FLOPs.

## Using `pdb` / `breakpoint()` inside your estimator

The interactive progress display can mask the debugger prompt when you drop a breakpoint inside `predict()`. Use one of the following patterns:

- **Recommended** — use `breakpoint()` rather than `pdb.set_trace()`. The CLI installs a hook that pauses the live display before the debugger starts, so the prompt appears cleanly:

  ```python
  def predict(self, mlp, budget):
      breakpoint()
      ...
  ```

- **With `pdb.set_trace()`** — pass `--no-rich` to disable the live display entirely:

  ```bash
  whest run --estimator ./my-estimator/estimator.py --runner local --no-rich
  ```

- **Or** set the standard env var before running:

  ```bash
  PYTHONBREAKPOINT=pdb.set_trace whest run --estimator ./... --runner local
  ```

  The CLI auto-detects this and switches to plain-text output automatically.

> Debugging is only supported with `--runner local`. The default `--runner server` isolates your estimator in a subprocess whose stdin/stdout carry the worker protocol, so interactive debuggers cannot attach there. Switch to `--runner local` (or `--runner inprocess`) when you want to break into your code.

## Next step

- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
- [Performance Tips](./performance-tips.md)
- [Manage Your FLOP Budget](./manage-flop-budget.md)

# Debugging Checklist

> [← Documentation](../README.md)

Use this page when your estimator runs but the score is bad, or something feels wrong. Work through the tiers in order.

## Tier 0: Pure-Python inner loop (fastest iteration)

For fast, no-framework iteration — e.g. to print intermediate activations, attach `pdb`, or sweep Monte Carlo sample counts — run your estimator as a plain Python script instead of going through `whest run`. The repo-root [`estimator.py`](../../estimator.py) is exactly this kind of self-contained loop: it constructs an MLP via `local_engine.build_mlp`, invokes the inline `Estimator`, and prints a FLOPs-vs-MSE convergence table. It's runnable two ways:

```bash
# 1) Direct: no CLI, no runner, no subprocess — just Python.
uv run python estimator.py

# 1b) Same file, with a side-by-side baseline comparison:
uv run python estimator.py --baseline mean_propagation

# 2) Scored via whestbench (same file, same class — honors BaseEstimator):
uv run whest run --estimator estimator.py
```

Edit `predict()` in `estimator.py` and re-run. See [Stage 1](../getting-started/stage-1-standalone.md) for the full walkthrough.

## Tier 1: Sanity checks (2 minutes)

Run validation:

```bash
whest validate --estimator estimator.py
```

If it fails, check:

- [ ] **Output shape:** does `predict()` return shape `(mlp.depth, mlp.width)`?
- [ ] **Finite values:** are all values finite? Check for `nan` or `inf` in your math.
- [ ] **Class name:** is your class named `Estimator`? The loader looks for this by default.

## Tier 2: Correctness checks (5 minutes)

Run your estimator and look at the report:

```bash
whest run --estimator estimator.py --n-mlps 3 --runner local --debug
```

Check:

- [ ] **Did `predict()` raise?** If `whest run` exits with status `1` and prints an "Estimator Errors" panel, your estimator raised an exception. Use `--debug` to include tracebacks inline in the panel, or add `--fail-fast` to halt at the first failure and let the raw Python traceback propagate.
- [ ] **Does zeros beat you?** If returning `fnp.zeros((mlp.depth, mlp.width))` scores better than your estimator, your predictions are wrong in a way that's worse than guessing zero.
- [ ] **Is `budget_exhausted` true?** If so, your estimator exceeded the FLOP budget and all predictions were zeroed. See [Manage Your FLOP Budget](./manage-flop-budget.md).
- [ ] **Are errors concentrated at deep layers?** Run with `--debug` and compare `all_layers_mse` — if early layers are good but later layers are bad, your propagation may accumulate errors.

## Tier 3: Optimization checks (10+ minutes)

Profile your FLOP usage:

```python
import flopscope as flops

with flops.BudgetContext(flop_budget=272_000_000_000) as budget:
    result = estimator.predict(mlp, budget=272_000_000_000)
    flops.budget_summary()
```

Check:

- [ ] **Is matmul dominant?** If >90% of FLOPs are in matmul, consider diagonal variance instead of full covariance.
- [ ] **Redundant computation?** Are you computing something in a loop that could be precomputed once?
- [ ] **Free operations wasted?** Remember: `fnp.zeros`, `fnp.transpose`, `fnp.reshape`, indexing cost 0 FLOPs.

## Using `pdb` / `breakpoint()` inside your estimator

The interactive progress display can mask the debugger prompt when you drop a breakpoint inside `predict()`. Use one of the following patterns:

- **Recommended** — use `breakpoint()` rather than `pdb.set_trace()`. The CLI installs a hook that pauses the live display before the debugger starts, so the prompt appears cleanly:

  ```python
  def predict(self, mlp, budget):
      breakpoint()
      ...
  ```

- **With `pdb.set_trace()`** — pass `--format plain` to disable the live display entirely:

  ```bash
  whest run --estimator estimator.py --runner local --format plain
  ```

- **Or** set the standard env var before running:

  ```bash
  PYTHONBREAKPOINT=pdb.set_trace whest run --estimator ./... --runner local
  ```

  The CLI auto-detects this and switches to plain output automatically.

> Debugging is best supported with `--runner local`. `--runner local` (or `--runner inprocess`) runs in-process for direct traces and interactive debugging. The isolation runners (`--runner subprocess`, legacy `--runner server`) communicate via worker protocol I/O, so interactive debuggers should be used in local mode.

## ➡️ Next step

- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
- [Performance Tips](./performance-tips.md)
- [Manage Your FLOP Budget](./manage-flop-budget.md)

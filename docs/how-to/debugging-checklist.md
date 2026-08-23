# Debugging checklist

> [← Documentation](../README.md)

Use this page when your estimator runs but the score is bad, or the results look wrong. Work through the tiers in order.

## Tier 0: Pure-Python inner loop (fastest iteration)

For fast, no-framework iteration (for example, to print intermediate activations, attach `pdb`, or sweep Monte Carlo sample counts), run your estimator as a plain Python script instead of going through `whest run`. The repo-root [`estimator.py`](../../estimator.py) is exactly this kind of self-contained loop: it constructs an MLP via `local_engine.build_mlp`, invokes the inline `Estimator`, and prints a FLOPs-vs-MSE convergence table. You can run it two ways:

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
uv run whest validate --estimator estimator.py
```

If it fails, check:

- [ ] **Output shape:** does `predict()` return shape `(mlp.depth, mlp.width)`, read off `mlp` and not a constant? The graded shape is `(16, 1024)`; a `(32, 256)` left over from Phase 1 fails every MLP. Every round's shape is in [Competition Rounds](../reference/rounds.md).
- [ ] **Finite values:** are all values finite? Check for `nan` or `inf` in your math.
- [ ] **Estimator class:** does the entrypoint file contain a class that subclasses `BaseEstimator`? whestbench finds a single subclass under any name; naming it `Estimator` (or passing `--class`) only matters when the file defines more than one.

## Tier 2: Correctness checks (5 minutes)

Run your estimator and look at the report:

```bash
uv run whest run --estimator estimator.py --n-mlps 3 --runner local --debug
```

Check:

- [ ] **Did `predict()` raise?** If `whest run` exits with status `1` and prints an "Estimator Errors" panel, your estimator raised an exception. To include tracebacks inline in the panel, use `--debug`; to halt at the first failure and let the raw Python traceback propagate, add `--fail-fast`.
- [ ] **Does zeros beat you?** If returning `fnp.zeros((mlp.depth, mlp.width))` scores better than your estimator, your predictions are wrong in a way that's worse than guessing zero.
- [ ] **Is `budget_exhausted` true?** If so, your estimator exceeded the FLOP budget and all predictions were zeroed. See [Manage Your FLOP Budget](./manage-flop-budget.md).
- [ ] **Is `residual_wall_time_exhausted` true?** To reproduce the grader's hard 400 ms residual cap, re-run with `--residual-wall-time-limit 0.4`. Any MLP over the cap is scored against zeros, so the score drops because Python plumbing got slower, not because the estimator got less accurate.
- [ ] **Are errors concentrated at deep layers?** Run with `--debug` and compare `all_layers_mse`. If early layers are good but later layers are bad, your propagation may accumulate errors.

## Tier 3: Optimization checks (10+ minutes)

Profile your FLOP usage. Save this as `profile_me.py` and run it with
`uv run python profile_me.py` from the repo root:

```python
import flopscope as flops
import local_engine
from estimator import Estimator          # or your own module

B_M = 2_199_023_255_552                  # 2**41 — the Phase 2 per-MLP budget
mlp = local_engine.build_mlp(width=1024, depth=16, seed=0)
estimator = Estimator()

with flops.BudgetContext(flop_budget=B_M) as budget:
    result = estimator.predict(mlp, B_M)
    print(budget.summary(by_namespace=True))   # by_namespace=True adds per-namespace attribution
```

Check:

- [ ] **Is one matrix op dominant?** If `matmul` or `einsum` is >90% of your FLOPs, consider diagonal variance instead of full covariance. At width 1024 the `einsum` in full covariance propagation is 99.66% of that estimator's FLOP total.
- [ ] **Redundant computation?** Are you computing something in a loop that could be precomputed once?
- [ ] **Free operations wasted?** Since flopscope 0.9 the free list is short: `fnp.zeros`, `fnp.empty`, no-copy views (`.T` / `fnp.transpose`, basic slicing, `fnp.asarray` on an existing flopscope array), and constructing an RNG. `reshape`, `ravel`, `stack`, `concatenate`, `ones`, `astype` and copies are all billed per element. See the [flopscope primer](../reference/flopscope-primer.md#operation-flop-costs).

## Using `pdb` / `breakpoint()` inside your estimator

The interactive progress display can mask the debugger prompt when you set a breakpoint inside `predict()`. Use one of the following patterns:

- **Recommended** — use `breakpoint()` rather than `pdb.set_trace()`. The CLI installs a hook that pauses the live display before the debugger starts, so the prompt appears cleanly:

  ```python
  def predict(self, mlp, budget):
      breakpoint()
      ...
  ```

- **With `pdb.set_trace()`** — pass `--format plain` to disable the live display entirely:

  ```bash
  uv run whest run --estimator estimator.py --runner local --format plain
  ```

- **Or** set the standard env var before running:

  ```bash
  PYTHONBREAKPOINT=pdb.set_trace uv run whest run --estimator ./... --runner local
  ```

  The CLI auto-detects this and switches to plain output automatically.

> Debug with `--runner local`. `--runner local` (or `--runner inprocess`) runs in-process for direct traces and interactive debugging. The isolation runners (`--runner subprocess`, legacy `--runner server`) communicate via worker protocol I/O, so use local mode whenever you need a debugger.

## ➡️ Next step

- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
- [Performance Tips](./performance-tips.md)
- [Manage Your FLOP Budget](./manage-flop-budget.md)

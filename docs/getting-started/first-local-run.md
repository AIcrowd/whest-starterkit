# First Local Run

## When to use this page

Use this page for your first end-to-end participant loop.

## Do this now

Create starter estimator files:

```bash
whest init ./my-estimator
```

Validate contract (fast sanity check — small fixed MLP, not a full evaluation):

```bash
whest validate --estimator ./my-estimator/estimator.py
```

Run local evaluation:

```bash
whest run --estimator ./my-estimator/estimator.py
```

For your first pass, this can be slower because a full local evaluation regenerates MLPs and recomputes ground-truth targets. A faster first debug loop is:

```bash
whest run --estimator ./my-estimator/estimator.py --n-mlps 1
```

Note: `whest run --estimator ...` defaults to `--runner local`.

Package a submission artifact:

```bash
whest package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

## ✅ Expected outcome

- validation succeeds,
- local run returns a report with `primary_score`,
- package command produces a `.tar.gz` artifact.

## 🛠 Common first failure

Symptom: validation fails because output shape or row count is wrong.

Fix: ensure `predict(mlp, budget)` returns exactly `mlp.depth` rows and each row is shape `(mlp.width,)`.

## 🧭 Debug runner flow (copy/paste)

### How to debug in order

1) Run with defaults (local runner, full MLP set):

```bash
whest run --estimator ./my-estimator/estimator.py
```

2) Add tracebacks to scoring failures:

```bash
whest run --estimator ./my-estimator/estimator.py --debug
```

3) Stop immediately at first exception to avoid scrolling:

```bash
whest run --estimator ./my-estimator/estimator.py --debug --fail-fast
```

4) If failures are coming from isolated mode, switch to local mode for direct traces:

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --debug
```

5) If you want interactive debugging in `predict()`, use local + plain output:

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --format plain
```

### What each runner means

- `local` is the default and executes your estimator in the current process.
- `inprocess` is the same as `local`.
- `subprocess` executes the estimator in a worker process for isolation.
- `server` is a legacy alias for `subprocess`.

### `--debug` and `--fail-fast` behavior

- `--debug` includes per-MLP tracebacks in the report and includes top-level tracebacks for setup/runtime command failures.
- `--fail-fast` stops scoring on the first unexpected `predict()` exception and exits immediately.
- Without `--fail-fast`, WhestBench still runs all MLPs, reports every failure, and exits non-zero at the end.

### Use `pdb` if you need it

Use `breakpoint()` (or `pdb.set_trace`) and force plain output so prompts are visible:

```python
def predict(self, mlp, budget):
    breakpoint()
    ...
```

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --format plain
```

With `pdb` specifically:

```bash
PYTHONBREAKPOINT=pdb.set_trace whest run --estimator ./my-estimator/estimator.py --runner local
```

The CLI auto-disables rich output for debugger sessions, so plain output is usually enough even without `--format plain`.

## ➡️ Next step

- [From Problem to Code](./from-problem-to-code.md) — complete walkthrough to build your first real estimator
- [Write an Estimator](../how-to/write-an-estimator.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
- [Use Evaluation Datasets](../how-to/use-evaluation-datasets.md) — skip repeated sampling when iterating
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
- [Whest Primer](../reference/whest-primer.md) — how FLOP tracking works under the hood

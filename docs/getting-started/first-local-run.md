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

When `run` fails, use this sequence:

1. Normal run (default local execution):

```bash
whest run --estimator ./my-estimator/estimator.py
```

2. Add traceback/debug fields:

```bash
whest run --estimator ./my-estimator/estimator.py --debug
```

3. If still unclear, switch to explicit in-process traceback fidelity:

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --debug
```

Why two runners:

- `local` (default): in-process execution with immediate startup, faster iteration, and full local tracebacks.
- `subprocess` (and legacy alias `server`): isolated process execution; use this when you need a stricter boundary.
- `docker`: use the dedicated evaluator container path when subprocess isolation is still not enough (not a local CLI flag in this repo yet).
- `inprocess`: alias for `local`.

Example failure and next command:

```text
Error [setup:SETUP_ERROR]: Estimator setup failed.
Use --debug to include a traceback.
Tip: For estimator-level tracebacks, rerun with --runner local --debug.
```

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --debug
```

## ➡️ Next step

- [From Problem to Code](./from-problem-to-code.md) — complete walkthrough to build your first real estimator
- [Write an Estimator](../how-to/write-an-estimator.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
- [Use Evaluation Datasets](../how-to/use-evaluation-datasets.md) — skip repeated sampling when iterating
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
- [Whest Primer](../reference/whest-primer.md) — how FLOP tracking works under the hood

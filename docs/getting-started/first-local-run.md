# First Local Run

## When to use this page

Use this page for your first end-to-end participant loop.

## Do this now

Create starter estimator files:

```bash
nestim init ./my-estimator
```

Validate contract (fast sanity check — small fixed MLP, not a full evaluation):

```bash
nestim validate --estimator ./my-estimator/estimator.py
```

Run local evaluation:

```bash
nestim run --estimator ./my-estimator/estimator.py
```

Note: `nestim run --estimator ...` defaults to `--runner subprocess`.

Package a submission artifact:

```bash
nestim package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
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

1. Normal run (default subprocess isolation):

```bash
nestim run --estimator ./my-estimator/estimator.py
```

2. Add traceback/debug fields:

```bash
nestim run --estimator ./my-estimator/estimator.py --debug
```

3. If still unclear, switch to in-process traceback fidelity:

```bash
nestim run --estimator ./my-estimator/estimator.py --runner inprocess --debug
```

Why two runners:

- `subprocess` (default): realistic isolation and safer runtime boundary.
- `inprocess`: easier estimator-level traceback debugging on your machine.

Example failure and next command:

```text
Error [setup:SETUP_ERROR]: Estimator setup failed.
Use --debug to include a traceback.
Tip: For estimator-level tracebacks, rerun with --runner inprocess --debug.
```

```bash
nestim run --estimator ./my-estimator/estimator.py --runner inprocess --debug
```

## ➡️ Next step

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
- [Use Evaluation Datasets](../how-to/use-evaluation-datasets.md) — skip repeated sampling when iterating
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)

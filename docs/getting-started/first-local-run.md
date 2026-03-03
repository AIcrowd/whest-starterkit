# First Local Run

## When to use this page

Use this page for your first end-to-end participant loop.

## Do this now

Create starter estimator files:

```bash
cestim init ./my-estimator
```

Validate stream contract (fast sanity check — small fixed circuit, not a full evaluation):

```bash
cestim validate --estimator ./my-estimator/estimator.py
```

Run local evaluation:

```bash
cestim run --estimator ./my-estimator/estimator.py
```

Note: `cestim run --estimator ...` defaults to `--runner subprocess`.

Package a submission artifact:

```bash
cestim package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

## ✅ Expected outcome

- validation succeeds,
- local run returns a report with `final_score`,
- package command produces a `.tar.gz` artifact.

## 🛠 Common first failure

Symptom: validation fails because output shape or row count is wrong.

Fix: ensure `predict(circuit, budget)` yields exactly `circuit.d` rows and each row is shape `(circuit.n,)`.

## 🧭 Debug runner flow (copy/paste)

When `run` fails, use this sequence:

1. Normal run (default subprocess isolation):

```bash
cestim run --estimator ./my-estimator/estimator.py
```

2. Add traceback/debug fields:

```bash
cestim run --estimator ./my-estimator/estimator.py --debug
```

3. If still unclear, switch to in-process traceback fidelity:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner inprocess --debug
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
cestim run --estimator ./my-estimator/estimator.py --runner inprocess --debug
```

## ➡️ Next step

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)

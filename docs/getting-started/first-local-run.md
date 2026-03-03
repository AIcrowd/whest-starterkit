# First Local Run

## 🚀 When to use this page

Use this page for your first end-to-end participant loop.

## Do this now

Create starter estimator files:

```bash
cestim init ./my-estimator
```

Validate stream contract:

```bash
cestim validate --estimator ./my-estimator/estimator.py
```

Run local evaluation:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner subprocess
```

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

## ➡️ Next step

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)

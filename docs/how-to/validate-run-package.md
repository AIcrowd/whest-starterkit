# Validate, Run, and Package

## When to use this page

Use this page for the standard local participant loop.

## Do this now

Validate estimator loading and stream contract:

> `cestim validate` is a fast sanity check using a small fixed circuit (width=4, depth=1). It verifies loading, stream shape, and value finiteness — not full behavioral or performance correctness. Always follow with `cestim run` for realistic tests.

```bash
cestim validate --estimator ./my-estimator/estimator.py
```

Run local scoring (recommended default runner):

```bash
cestim run --estimator ./my-estimator/estimator.py
```

`cestim run` defaults to `--runner subprocess`.

Run faster local debug path:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner inprocess
```

Run with machine-readable output:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner subprocess --json
```

Package submission artifact:

```bash
cestim package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

Optional files during packaging:

```bash
cestim package \
  --estimator ./my-estimator/estimator.py \
  --requirements ./my-estimator/requirements.txt \
  --submission-metadata ./my-estimator/submission.yaml \
  --approach ./my-estimator/APPROACH.md \
  --output ./submission.tar.gz
```

## ✅ Expected outcome

- `validate` passes,
- `run` produces a score report,
- `package` creates a `.tar.gz` artifact.

## 🛠 Common first failure

Symptom: `run` fails after `validate` passed.

Use this escalation flow:

1. Retry with debug info in default subprocess mode:

```bash
cestim run --estimator ./my-estimator/estimator.py --debug
```

2. If traceback still feels opaque, rerun in-process:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner inprocess --debug
```

Runner tradeoff:

- `subprocess` (default): realistic isolation, safer runtime boundary.
- `inprocess`: better local traceback fidelity while debugging estimator code.

Concrete example:

```text
Error [predict:PREDICT_ERROR]: Estimator predict failed.
Use --debug to include a traceback.
Tip: For estimator-level tracebacks, rerun with --runner inprocess --debug.
```

Next command to run:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner inprocess --debug
```

## ➡️ Next step

- [CLI Reference](../reference/cli-reference.md)
- [Score Report Fields](../reference/score-report-fields.md)

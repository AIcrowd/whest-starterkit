# Validate, Run, and Package

## 🚀 When to use this page

Use this page for the standard local participant loop.

## Do this now

Validate estimator loading and stream contract:

```bash
cestim validate --estimator ./my-estimator/estimator.py
```

Run local scoring (recommended default runner):

```bash
cestim run --estimator ./my-estimator/estimator.py --runner subprocess
```

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

Fix: run again with `--json --debug` to inspect structured error stage/code, then check row shape/count and non-finite values first.

## ➡️ Next step

- [CLI Reference](../reference/cli-reference.md)
- [Score Report Fields](../reference/score-report-fields.md)

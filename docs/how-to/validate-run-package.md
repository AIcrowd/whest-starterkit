# Validate, Run, and Package

## When to use this page

Use this page for the standard local participant loop.

## Do this now

Validate estimator loading and output contract:

> `whest validate` is a fast sanity check using a small fixed MLP (width=4, depth=2). It verifies loading, output shape, and value finiteness — not full behavioral or performance correctness. Always follow with `whest run` for realistic tests.

```bash
whest validate --estimator ./my-estimator/estimator.py
```

Run local scoring (recommended default runner):

```bash
whest run --estimator ./my-estimator/estimator.py
```

`whest run` defaults to `--runner local` for fast iteration.

Run against a pre-created dataset (skips sampling — much faster for repeated runs):

```bash
whest create-dataset -o my_dataset.npz
whest run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz
```

See [Use Evaluation Datasets](./use-evaluation-datasets.md) for details.

Run faster local debug path:

```bash
whest run --estimator ./my-estimator/estimator.py --runner local
```

Run with machine-readable output:

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --format json
```

`--json` still works as an alias, but `--format rich|plain|json` is the canonical output selector across the CLI.

Package submission artifact:

```bash
whest package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

Optional files during packaging:

```bash
whest package \
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

1. Retry with debug info:

```bash
whest run --estimator ./my-estimator/estimator.py --debug
```

2. If traceback still feels opaque, rerun in-process:

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --debug
```

For runner modes and the debug escalation ladder, see [First Local Run — Debug Runner Flow](../getting-started/first-local-run.md#-debug-runner-flow-copypaste).

Concrete example:

```text
Error [predict:PREDICT_ERROR]: Estimator predict failed.
Use --debug to include a traceback.
Tip: For estimator-level tracebacks, rerun with --runner local --debug.
```

Next command to run:

```bash
whest run --estimator ./my-estimator/estimator.py --runner local --debug
```

## ➡️ Next step

- [Use Evaluation Datasets](./use-evaluation-datasets.md)
- [CLI Reference](../reference/cli-reference.md)
- [Score Report Fields](../reference/score-report-fields.md)

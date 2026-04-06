# Validate, Run, and Package

## When to use this page

Use this page for the standard local participant loop.

## Do this now

Validate estimator loading and output contract:

> `nestim validate` is a fast sanity check using a small fixed MLP (width=4, depth=2). It verifies loading, output shape, and value finiteness — not full behavioral or performance correctness. Always follow with `nestim run` for realistic tests.

```bash
nestim validate --estimator ./my-estimator/estimator.py
```

Run local scoring (recommended default runner):

```bash
nestim run --estimator ./my-estimator/estimator.py
```

`nestim run` defaults to `--runner server`.

Run against a pre-created dataset (skips sampling — much faster for repeated runs):

```bash
nestim create-dataset -o my_dataset.npz
nestim run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz
```

See [Use Evaluation Datasets](./use-evaluation-datasets.md) for details.

Run faster local debug path:

```bash
nestim run --estimator ./my-estimator/estimator.py --runner local
```

Run with machine-readable output:

```bash
nestim run --estimator ./my-estimator/estimator.py --runner server --json
```

Package submission artifact:

```bash
nestim package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

Optional files during packaging:

```bash
nestim package \
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

1. Retry with debug info in default server mode:

```bash
nestim run --estimator ./my-estimator/estimator.py --debug
```

2. If traceback still feels opaque, rerun in-process:

```bash
nestim run --estimator ./my-estimator/estimator.py --runner local --debug
```

Runner tradeoff:

- `server` (default): realistic isolation — your estimator runs against the mechestim server.
- `local`: in-process execution with better traceback fidelity while debugging.

Concrete example:

```text
Error [predict:PREDICT_ERROR]: Estimator predict failed.
Use --debug to include a traceback.
Tip: For estimator-level tracebacks, rerun with --runner local --debug.
```

Next command to run:

```bash
nestim run --estimator ./my-estimator/estimator.py --runner local --debug
```

## ➡️ Next step

- [Use Evaluation Datasets](./use-evaluation-datasets.md)
- [CLI Reference](../reference/cli-reference.md)
- [Score Report Fields](../reference/score-report-fields.md)

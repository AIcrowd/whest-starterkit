# Validate, Run, and Package

## When To Use This Page

Use this page for the standard local participant loop.

## Validate

```bash
cestim validate --estimator ./my-estimator/estimator.py
```

## Run

Safer isolation boundary (recommended default):

```bash
cestim run --estimator ./my-estimator/estimator.py --runner subprocess
```

Faster local debugging path:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner inprocess
```

Structured output for automation:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner subprocess --json
```

## Package

```bash
cestim package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

Optional files:

```bash
cestim package \
  --estimator ./my-estimator/estimator.py \
  --requirements ./my-estimator/requirements.txt \
  --submission-metadata ./my-estimator/submission.yaml \
  --approach ./my-estimator/APPROACH.md \
  --output ./submission.tar.gz
```

## Next

- [CLI Reference](../reference/cli-reference.md)
- [Score Report Fields](../reference/score-report-fields.md)

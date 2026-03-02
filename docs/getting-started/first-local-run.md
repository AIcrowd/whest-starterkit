# First Local Run

## When To Use This Page

Use this page for your first end-to-end participant loop.

## Steps

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

## Expected Outcome

- validation succeeds,
- local run returns a report with `final_score`,
- package command produces a `.tar.gz` artifact.

## Next

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)

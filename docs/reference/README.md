# Reference — Exact contracts and APIs

> [← Documentation](../README.md)

Lookup material. No tutorials here. Go to [How-to](../how-to/) for guidance, [Concepts](../concepts/) for the why, [Tutorial](../getting-started/) for the climbing trail.

## The round

| Doc | What it covers |
|---|---|
| [rounds.md](rounds.md) | Every round the challenge has run — shape, budget, wall caps, and how residual time was charged — side by side. Start here if a number you read somewhere else does not match what your local run reports, or if you need to reproduce a score from an earlier round. |

## Estimator API

| Doc | What it covers |
|---|---|
| [estimator-contract.md](estimator-contract.md) | The Phase 2 limits table, required `predict(mlp, budget)` signature, optional `setup`/`teardown` lifecycle, `SetupContext` field reference, output requirements, what your submission is allowed to import, and a failure-semantics table mapping every failure mode to the report field that surfaces it. |
| [code-patterns.md](code-patterns.md) | `flopscope` patterns: what's free, what costs, the rectified-Gaussian first-moment derivation, and the regimes where the Gaussian assumption breaks. |
| [local-engine-api.md](local-engine-api.md) | The `local_engine.py` factory and Monte-Carlo helpers used in Stage 1, with their Phase 2 defaults and measured sampling costs. |

## FLOP and scoring details

| Doc | What it covers |
|---|---|
| [flopscope-primer.md](flopscope-primer.md) | `BudgetContext` ownership across stages, complete public-attribute reference (`flops_used`, `flops_remaining`, `wall_time_s`, …), and the op cost table. |
| [score-report-fields.md](score-report-fields.md) | Every field you'll see in `whest run` output, with interpretation guidance and a note on which fields are Phase 1 leftovers. |

## CLI

| Doc | What it covers |
|---|---|
| [cli-reference.md](cli-reference.md) | Which `whest` subcommand belongs to which stage, the daily submission quota, and the `whest run` limit and dataset flags with their 0.16.0 defaults. Full syntax lives upstream. |
| [whest-doctor.md](whest-doctor.md) | Each `whest doctor` health check, what it verifies, and how to fix a `WARN` or `FAIL` row. |

## ➡️ Where to look next

- Trying to do something and need a procedure? → [How-to](../how-to/).
- Curious *why* the contract is shaped this way? → [Concepts](../concepts/).

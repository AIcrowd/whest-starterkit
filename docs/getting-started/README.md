# Tutorial — The 6-stage ladder

> [← Documentation](../README.md)

The tutorial trail. Each stage is a single command on the same `estimator.py`, with the harness adding one more level of formality at each step. Read top-to-bottom: Stage 4's subprocess isolation catches bugs that Stage 3 hides, so skipping ahead is rarely worth it.

| Stage | Command | What it adds | Doc |
|---|---|---|---|
| 1 | `uv run python estimator.py` | The math. Iterate locally with `flopscope` and `local_engine.py`; no `whest` CLI required. | [stage-1-standalone.md](stage-1-standalone.md) |
| 2 | `uv run whest validate --estimator estimator.py` | Contract correctness — class resolved, optional `setup()` runs, shape, finite values. | [stage-2-validate.md](stage-2-validate.md) |
| 3 | `uv run whest run --estimator estimator.py --runner local` | Real scoring against the grader's MLP suite, in-process (so `pdb` works). | [stage-3-run-local.md](stage-3-run-local.md) |
| 4 | `uv run whest run --estimator estimator.py --runner subprocess` | Subprocess isolation — catches state-bleed between MLPs, dirty imports, RNG re-use. | [stage-4-run-subprocess.md](stage-4-run-subprocess.md) |
| 5 | `uv run whest run --estimator estimator.py --runner docker` | Production-equivalent grader environment. **Coming soon.** | [stage-5-run-docker.md](stage-5-run-docker.md) |
| 6 | `uv run whest package --estimator estimator.py -o submission.tar.gz` | Package the submission tarball for AIcrowd. | [stage-6-package.md](stage-6-package.md) |

Each stage doc carries an "Expected outcome" callout so you know what success looks like before climbing — and a "Ladder" strip at the top so you always know where you are.

## ➡️ Where to look next

- Ready to ship? → [Stage 6 → Submit to AIcrowd](stage-6-package.md#-submit-to-aicrowd) (`whest login` then `whest submit`).
- Got a working estimator and want a better score? → [How-to: algorithm ideas](../how-to/algorithm-ideas.md), [Reference: code patterns](../reference/code-patterns.md).
- Score regressed after a change? → [How-to: debugging checklist](../how-to/debugging-checklist.md), [Troubleshooting](../troubleshooting/).
- Need the exact contract? → [Reference: estimator contract](../reference/estimator-contract.md).
- Sanity-check before clicking "submit"? → [How-to: pre-submission checklist](../how-to/pre-submission-checklist.md).

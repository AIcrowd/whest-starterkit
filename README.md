<div align="center">
  <img src="assets/logo/logo.png" alt="ARC Whitebox Estimation Challenge logo" style="height: 120px;">
</div>

# Whitebox Estimation — Starter Kit

[![CI](https://github.com/AIcrowd/whest-starterkit/actions/workflows/ci.yml/badge.svg)](https://github.com/AIcrowd/whest-starterkit/actions/workflows/ci.yml)
[![whestbench pin](https://github.com/AIcrowd/whest-starterkit/actions/workflows/bump-whestbench.yml/badge.svg)](https://github.com/AIcrowd/whest-starterkit/actions/workflows/bump-whestbench.yml)

<div align="center">
  <img src="assets/demo.gif" alt="whest-starterkit first 5 minutes" width="720">
</div>

## 60-Second Overview

You are given a randomly-initialized ReLU MLP and a FLOP budget. Your job: predict the per-neuron mean activation under N(0, 1) input, **without** running anywhere near the budget's worth of forward passes. The lower your MSE against the ground-truth Monte-Carlo means (per FLOP spent), the better your score.

This kit walks you up a "ladder of formality" — start by iterating math locally with zero CLI knowledge, then graduate to the harness when you're ready.

## Your First 5 Minutes (Stage 1: just `whest`)

```bash
git clone https://github.com/AIcrowd/whest-starterkit.git
cd whest-starterkit
```

```bash-test
uv sync && uv run python estimator.py
```

You just ran an estimator and saw it converge against ground truth. Now edit `predict()` in [estimator.py](estimator.py) and re-run.

Want to compare against a reference?

```bash-test
uv run python estimator.py --baseline mean_propagation
```

## Try the Examples (still Stage 1)

```bash-test
uv run python examples/02_mean_propagation.py
uv run python examples/03_covariance_propagation.py
uv run python examples/04_combined.py
```

See [examples/README.md](examples/README.md) for the curriculum table.

## Climb the Ladder (Stages 2-6)

| Stage | Command | What it adds | Walkthrough |
|---|---|---|---|
| 1 | `uv run python estimator.py` | The math. Estimator vs Monte Carlo. | [Standalone &rarr;](docs/getting-started/stage-1-standalone.md) |
| 2 | `uv run whest validate --estimator estimator.py` | Contract correctness (shapes, types). | [Validate &rarr;](docs/getting-started/stage-2-validate.md) |
| 3 | `uv run whest run --estimator estimator.py --runner local` | Real scoring, in-process, debuggable with `pdb`. | [Run local &rarr;](docs/getting-started/stage-3-run-local.md) |
| 4 | `uv run whest run --estimator estimator.py --runner subprocess` | Isolation; closer to grader environment. | [Subprocess &rarr;](docs/getting-started/stage-4-run-subprocess.md) |
| 5 | `uv run whest run --estimator estimator.py --runner docker` | **Coming soon.** Production-equivalent grader. | [Docker &rarr;](docs/getting-started/stage-5-run-docker.md) |
| 6 | `uv run whest package --estimator estimator.py -o submission.tar.gz` | Submission artifact. | [Package &rarr;](docs/getting-started/stage-6-package.md) |

## When Something Breaks

```bash-test
uv run whest doctor
```

Or check [docs/troubleshooting/](docs/troubleshooting/).

## Repo Layout

```
.
├── estimator.py          ← Edit this. Stages 1-6 all use it.
├── local_engine.py       ← Pedagogical re-implementation; iterate freely.
├── examples/             ← Reference estimators 01-04.
├── docs/                 ← Stage-by-stage walkthroughs, concepts, reference.
└── tests/                ← Drift gates (README commands + local_engine parity).
```

## License & Contributing

See [LICENSE](LICENSE) and [docs/RELEASING.md](docs/RELEASING.md).

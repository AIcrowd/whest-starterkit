<div align="center">
  <img src="assets/logo/logo.png" alt="ARC Whitebox Estimation Challenge logo" style="height: 120px;">
</div>

# Whitebox Estimation — Starter Kit

[![CI](https://github.com/AIcrowd/whest-starterkit/actions/workflows/ci.yml/badge.svg)](https://github.com/AIcrowd/whest-starterkit/actions/workflows/ci.yml)

<div align="center">
  <img src="assets/demo.gif" alt="whest-starterkit first 5 minutes" width="720">
</div>

## 🎬 60-Second Overview

You are given a randomly-initialized ReLU MLP and a FLOP budget. Predict the per-neuron mean activation under N(0, 1) input — without running anywhere near the budget's worth of forward passes. Your score is the MSE against the ground-truth Monte-Carlo means: lower is better.

<div align="center">
  <img src="assets/whestbench-explorer-visualization.svg" alt="A small ReLU MLP (width 4, depth 5) shown as a layer-by-layer heatmap of per-neuron mean activations after Monte-Carlo ground-truth estimation; rows are layers, columns are neurons, color intensity is mean activation" width="720">
  <br>
  <sub><em>Per-neuron mean activations of a small MLP (width 4, depth 5) after Monte-Carlo ground truth — exactly what your estimator predicts. Generate your own at the <a href="https://aicrowd.github.io/whestbench-explorer/">hosted WhestBench Explorer</a>.</em></sub>
</div>

The kit is structured as a six-stage **ladder of formality**: each stage adds one more layer of harness rigor. Start at Stage 1 (pure local math, zero CLI knowledge); climb to Stage 6 (a packaged submission) when you're ready.

## 🚀 Your First 5 Minutes (Stage 1: just `python`)

```bash
git clone https://github.com/AIcrowd/whest-starterkit.git
cd whest-starterkit
```

```bash-test
uv sync && uv run python estimator.py
```

That run printed a Monte-Carlo convergence table — your estimator's predictions, the FLOPs it used, and the MSE against ground truth at increasing sample counts. To experiment, edit `predict()` in [estimator.py](estimator.py) and re-run.

Compare against a bundled baseline:

```bash-test
uv run python estimator.py --baseline mean_propagation
```

## 🧪 Try the Examples (still Stage 1)

```bash-test
uv run python examples/02_mean_propagation.py
uv run python examples/03_covariance_propagation.py
```

See [examples/README.md](examples/README.md) for the curriculum table.

## 🪜 Climb the Ladder (Stages 2-6)

| Stage | Command | What it adds |
|---|---|---|
| [1: Iterate locally](docs/getting-started/stage-1-standalone.md) | `uv run python estimator.py` | The math. Estimator vs Monte Carlo. |
| [2: Validate the contract](docs/getting-started/stage-2-validate.md) | `uv run whest validate --estimator estimator.py` | Contract correctness (shapes, types). |
| [3: Run locally](docs/getting-started/stage-3-run-local.md) | `uv run whest run --estimator estimator.py --runner local` | Real scoring, in-process, debuggable with `pdb`. |
| [4: Subprocess runner](docs/getting-started/stage-4-run-subprocess.md) | `uv run whest run --estimator estimator.py --runner subprocess` | Isolation; closer to grader environment. |
| [5: Docker runner](docs/getting-started/stage-5-run-docker.md) | `uv run whest run --estimator estimator.py --runner docker` | **Coming soon.** Production-equivalent grader. |
| [6: Package your submission](docs/getting-started/stage-6-package.md) | `uv run whest package --estimator estimator.py -o submission.tar.gz` | Submission artifact. |

## 🏁 Submit to AIcrowd

Climbed to Stage 6? Ship it from the CLI. Log in once with your
[AIcrowd API key](https://www.aicrowd.com/participants/me/customize):

```bash
uv run whest login
```

Then package + submit in one step (add `--watch` to follow it to a score):

```bash
uv run whest submit --estimator estimator.py --watch
```

Your score and per-MLP detail land on the
[challenge leaderboard](https://www.aicrowd.com/). Full walkthrough:
[Stage 6 → Submit to AIcrowd](docs/getting-started/stage-6-package.md#-submit-to-aicrowd).

## 🚑 When Something Breaks

```bash-test
uv run whest doctor
```

Reads as a 6-row health check; see [docs/reference/whest-doctor.md](docs/reference/whest-doctor.md) for what each row means and how to fix warnings.

Or check [docs/troubleshooting/](docs/troubleshooting/).

## 📚 Documentation

Past Stage 1, the documentation is organized into six sections — pick whichever matches your task. Full map and guided reading paths at **[docs/](docs/README.md)**.

<details>
<summary>🪜 <b><a href="docs/getting-started/">Tutorial</a></b> — Climb the 6-stage ladder above</summary>

- [Stage 1: Iterate locally](docs/getting-started/stage-1-standalone.md) — The math; `flopscope` + `local_engine.py`, no `whest` CLI.
- [Stage 2: Validate the contract](docs/getting-started/stage-2-validate.md) — Class resolves, `setup()` runs, shape, finite values.
- [Stage 3: Run locally](docs/getting-started/stage-3-run-local.md) — Real scoring against the grader's MLP suite, in-process.
- [Stage 4: Subprocess runner](docs/getting-started/stage-4-run-subprocess.md) — Catches state-bleed, RNG re-use, dirty imports.
- [Stage 5: Docker runner](docs/getting-started/stage-5-run-docker.md) — Production-equivalent grader env. **Coming soon.**
- [Stage 6: Package your submission](docs/getting-started/stage-6-package.md) — Build the AIcrowd submission tarball.

</details>

<details>
<summary>📖 <b><a href="docs/concepts/">Concepts</a></b> — Why this challenge exists, what's measured, how ground truth works</summary>

- [Problem Setup](docs/concepts/problem-setup.md) — MLP architecture, He init, the research question, further reading.
- [Scoring Model](docs/concepts/scoring-model.md) — Pipeline diagram, `adjusted_final_layer_score` / `all_layers_mse` formulas, calibration table.
- [Ground Truth](docs/concepts/ground-truth.md) — How the evaluator computes reference values via Monte Carlo.

</details>

<details>
<summary>🔧 <b><a href="docs/how-to/">How-to</a></b> — Recipes: write, debug, optimize, submit</summary>

**Writing and iterating**
- [Write an Estimator](docs/how-to/write-an-estimator.md) — Minimal structure, contract checklist, common first failure.
- [Inspect MLP Structure](docs/how-to/inspect-mlp-structure.md) — Traversing the `MLP` object.
- [Validate, Run, Package](docs/how-to/validate-run-package.md) — The standard local loop, plus a useful-flags table.
- [Use Evaluation Datasets](docs/how-to/use-evaluation-datasets.md) — Pre-create datasets for fast, reproducible iteration.

**Optimizing**
- [Algorithm Ideas](docs/how-to/algorithm-ideas.md) — Monte Carlo, mean propagation, covariance, hybrid, plus open directions.
- [Manage FLOP Budget](docs/how-to/manage-flop-budget.md) — Where your FLOPs go; line-by-line walkthrough of `examples/02`.
- [Performance Tips](docs/how-to/performance-tips.md) — Matmul placement, free ops, env-var knobs.

**Debugging and shipping**
- [Debugging Checklist](docs/how-to/debugging-checklist.md) — Tiered procedure when something feels wrong.
- [Pre-Submission Checklist](docs/how-to/pre-submission-checklist.md) — One-screen gate before you click submit.

</details>

<details>
<summary>📚 <b><a href="docs/reference/">Reference</a></b> — Exact contracts, schemas, lookup material</summary>

**Estimator API**
- [Estimator Contract](docs/reference/estimator-contract.md) — `predict`/`setup`/`teardown` signatures, `SetupContext`, failure-semantics table, lifecycle diagram.
- [Code Patterns](docs/reference/code-patterns.md) — `flopscope` patterns, ReLU expectation derivation, when the Gaussian assumption breaks.
- [<code>local_engine</code> API](docs/reference/local-engine-api.md) — Stage 1's MLP factory and Monte-Carlo helpers.

**FLOP and scoring details**
- [Flopscope Primer](docs/reference/flopscope-primer.md) — `BudgetContext` ownership, attribute reference, op cost table.
- [Score Report Fields](docs/reference/score-report-fields.md) — Every field you'll see in `whest run` output.

**CLI**
- [CLI Reference](docs/reference/cli-reference.md) — Pointer at the upstream `whest` CLI.
- [<code>whest doctor</code>](docs/reference/whest-doctor.md) — The 6 install/env checks and how to fix WARN/FAIL rows.

</details>

<details>
<summary>🚑 <b><a href="docs/troubleshooting/">Troubleshooting</a></b> — When something breaks</summary>

- [Common Participant Errors](docs/troubleshooting/common-participant-errors.md) — Symptom → cause → fix-now → verify.
- [FAQ](docs/troubleshooting/faq.md) — Quick answers; includes "local score great, submission 10x worse".

</details>

<details>
<summary>🔬 <b><a href="docs/advanced/">Advanced</a></b> — Deeper tooling</summary>

- [Profile Simulation](docs/advanced/profile-simulation.md) — FLOP and time breakdown of your `predict()` call.
- [WhestBench Explorer](docs/advanced/use-whestbench-explorer.md) — Hosted interactive visualizer at [aicrowd.github.io/whestbench-explorer](https://aicrowd.github.io/whestbench-explorer/) for inspecting MLPs and ground truth.

</details>

## 📁 Repo Layout

```
├── estimator.py     ← The participant's entry point; every stage operates on this file.
├── local_engine.py  ← Single-file re-implementation of the harness; safe to read end-to-end.
├── examples/        ← Numbered reference estimators (01–03) with a curriculum table.
├── docs/            ← Full documentation; start at docs/README.md.
└── tests/           ← Drift gates: README commands + local_engine parity.
```

## ⚖️ License & Contributing

See [LICENSE](LICENSE) and [docs/RELEASING.md](docs/RELEASING.md).

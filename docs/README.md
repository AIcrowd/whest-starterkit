# Documentation

The starter kit teaches you to write a Python estimator that predicts
per-neuron mean activations of a randomly-initialized ReLU MLP under a
FLOP budget. **Lower MSE per FLOP wins.** New here? See the [60-second
overview](../README.md#60-second-overview) in the root README, then
jump in below.

## 🗺️ Map

|     | Section | What lives here | Start with |
|---|---|---|---|
| 🪜 | **[Tutorial](getting-started/)** | The 5-stage ladder. Run code, climb step by step. | [Stage 1: Iterate locally](getting-started/stage-1-standalone.md) |
| 📖 | **[Concepts](concepts/)** | Why the challenge exists, what's being measured, how the math works. | [Problem Setup](concepts/problem-setup.md) |
| 🔧 | **[How-to](how-to/)** | Recipes for specific tasks: write, debug, optimize, submit. | [Write an Estimator](how-to/write-an-estimator.md) |
| 📚 | **[Reference](reference/)** | Exact contracts, field schemas, CLI options, attribute lists. | [Estimator Contract](reference/estimator-contract.md) |
| 🚑 | **[Troubleshooting](troubleshooting/)** | When something breaks. | [Common Participant Errors](troubleshooting/common-participant-errors.md) |
| 🔬 | **[Advanced](advanced/)** | Profiler and the hosted Explorer visualizer. | [Profile Simulation](advanced/profile-simulation.md) |

## 🧭 Reading paths

Pick the line that sounds like you.

### "I just cloned the kit."

[Stage 1](getting-started/stage-1-standalone.md) → skim [scoring-model](concepts/scoring-model.md) so you know what's being measured → [Stage 2](getting-started/stage-2-validate.md) → [write-an-estimator](how-to/write-an-estimator.md) → [Stage 3](getting-started/stage-3-run-local.md).

### "I have a working estimator. I want a better score."

[algorithm-ideas](how-to/algorithm-ideas.md) → [code-patterns](reference/code-patterns.md) (the ReLU expectation and where it breaks) → [manage-flop-budget](how-to/manage-flop-budget.md) (the FLOP walkthrough table) → [advanced/profile-simulation](advanced/profile-simulation.md).

### "My score regressed after a change."

[debugging-checklist](how-to/debugging-checklist.md) → [common-participant-errors](troubleshooting/common-participant-errors.md) → [FAQ: local vs remote score mismatch](troubleshooting/faq.md#my-local-score-is-great-but-my-submission-scores-10x-worse--why) → [whest-doctor](reference/whest-doctor.md).

### "I'm about to submit."

[pre-submission-checklist](how-to/pre-submission-checklist.md) → [Stage 5](getting-started/stage-5-package.md) → [score-report-fields](reference/score-report-fields.md) (so you can read the leaderboard report).

## 🪜 By Stage (the ladder)

1. [Stage 1: Iterate locally](getting-started/stage-1-standalone.md)
2. [Stage 2: Validate the contract](getting-started/stage-2-validate.md)
3. [Stage 3: Run locally](getting-started/stage-3-run-local.md)
4. [Stage 4: Subprocess runner](getting-started/stage-4-run-subprocess.md)
5. [Stage 5: Package your submission](getting-started/stage-5-package.md)

## 🎯 By Need

| I want to... | Read |
|---|---|
| Understand the math | [concepts/problem-setup.md](concepts/problem-setup.md), [concepts/scoring-model.md](concepts/scoring-model.md), [concepts/ground-truth.md](concepts/ground-truth.md) |
| Know what success looks like at each stage | the **Expected outcome** callout at the bottom of each [stage doc](getting-started/), and the [example benchmarks](concepts/scoring-model.md#example-estimator-benchmarks) |
| Improve my score (0.5 → 0.005) | [how-to/algorithm-ideas.md](how-to/algorithm-ideas.md), [how-to/manage-flop-budget.md](how-to/manage-flop-budget.md), [how-to/performance-tips.md](how-to/performance-tips.md) |
| Push from 0.005 → 0.0005 | [reference/code-patterns.md](reference/code-patterns.md) (ReLU expectation, when it breaks), [how-to/algorithm-ideas.md#open-directions](how-to/algorithm-ideas.md#open-directions), [advanced/profile-simulation.md](advanced/profile-simulation.md) |
| Debug a broken estimator | [troubleshooting/common-participant-errors.md](troubleshooting/common-participant-errors.md), [troubleshooting/faq.md](troubleshooting/faq.md), [how-to/debugging-checklist.md](how-to/debugging-checklist.md) |
| Sanity-check before submitting | [how-to/pre-submission-checklist.md](how-to/pre-submission-checklist.md) |
| Look up the precise contract | [reference/estimator-contract.md](reference/estimator-contract.md), [reference/score-report-fields.md](reference/score-report-fields.md), [reference/local-engine-api.md](reference/local-engine-api.md), [reference/cli-reference.md](reference/cli-reference.md) |
| Diagnose my install / environment | [reference/whest-doctor.md](reference/whest-doctor.md) |
| Profile or visualize | [advanced/use-whestbench-explorer.md](advanced/use-whestbench-explorer.md), [advanced/profile-simulation.md](advanced/profile-simulation.md) |

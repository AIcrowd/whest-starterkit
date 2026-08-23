# Concepts — why this challenge exists

> [← Documentation](../README.md)

Background reading. These docs explain the problem framing, the scoring metric, how ground truth is generated, and what code a submission is allowed to use. Helpful before you start tuning; essential before debating leaderboard outcomes.

| Doc | What it covers |
|---|---|
| [problem-setup.md](problem-setup.md) | The MLP architecture (width 1024, depth 16 this round), He initialization, and the research framing: why "competing with sampling" is the milestone this challenge targets. Includes a "Further reading" pointer to the relevant ARC posts and papers. |
| [scoring-model.md](scoring-model.md) | How the leaderboard score is computed: ASCII pipeline diagram, explicit equation block for `adjusted_final_layer_score` and `all_layers_mse`, what happens when a cap is blown (the FLOP budget, the 400 ms residual cap, the 120 s wall clock, or the 5 s `setup()` timeout), and measured cost/accuracy tables from the bundled examples. |
| [ground-truth.md](ground-truth.md) | How the evaluator generates the reference values you're scored against: Monte-Carlo sampling, sample counts, the inherent noise floor. |
| [allowed-code.md](allowed-code.md) | What a submission may use (the grader's interpreter, the flopscope client API, the pure-Python stdlib), the prohibition list, the data-file carve-out, what residual wall time is for, and how the rule is enforced. |

> **Phase 2 changed the scoring model.** Compute is now the analytical FLOP count and nothing else (`C_m = F_m`); residual wall-time is a hard 400 ms cap per MLP rather than something you can pay for. If you competed in Phase 1, re-read [scoring-model.md](scoring-model.md) rather than assuming.

Read in order if you want the full picture. **At minimum, skim `scoring-model.md`.** It defines the metric the leaderboard ranks on.

## ➡️ Where to look next

- Ready to write code? → [Tutorial: Stage 1](../getting-started/stage-1-standalone.md), [How-to: write-an-estimator](../how-to/write-an-estimator.md).
- Want the exact API contract? → [Reference: estimator-contract](../reference/estimator-contract.md).

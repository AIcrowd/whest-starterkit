# Competition rounds

> [← Reference](README.md)

The challenge has run three rounds. Each one has its own MLP shape, FLOP budget, and
rules for how wall time is charged. All three are kept side by side rather than
replaced, so a score from an earlier round can still be reproduced under the rulebook
it actually ran against.

**Current round:** you are being scored under `v2-phase2` — width 1024,
depth 16, `2**41` FLOPs per MLP, `C_m = F_m`.

## The round you are being scored under

```text
tag                          v2-phase2
shape                        width 1024 x depth 16   ->  predict() returns (16, 1024)
flop_budget          B_m     2**41 = 2,199,023,255,552
effective compute    C_m     C_m = F_m               (lambda = 0; residual time is not priced)
residual wall time           capped at 0.4 s per MLP -- crossing it fails that MLP
wall time per predict()      120 s
setup() timeout              5 s, and exceeding it fails the whole submission
ground-truth samples N       1e9 per MLP, baked into the dataset
```

Since whestbench 0.16.0 every one of those is the `whest run` **default**, so a plain
local run with no flags applies the Phase 2 parameters.

## Every round, side by side

| | `v1-warmup` | `v1-phase1` | `v2-phase2` **(current)** |
|---|---|---|---|
| **Dataset tag** | `v1-warmup` | `v1-phase1` | **`v2-phase2`** |
| **Shape** (width × depth) | 256 × 8 | 256 × 32 | **1024 × 16** |
| **`predict()` returns** | `(8, 256)` | `(32, 256)` | **`(16, 1024)`** |
| **One forward pass** (`2·d·w²`) | 1,048,576 FLOPs | 4,194,304 FLOPs | **33,554,432 FLOPs** |
| **Budget `B_m`** | 6.8e10 | 2.72e11 | **2⁴¹ = 2,199,023,255,552** |
| **Budget ÷ forward pass** | ≈ 64,850 | ≈ 64,850 | **65,536** |
| **Ground-truth samples `N`** | 1e9 | 1e9 | 1e9 |
| **λ** (residual rate) | 1e11 | 1e11 | **0 — deprecated** |
| **Residual cap** | none | none | **0.4 s** |
| **Wall cap per `predict()`** | 60 s | 60 s | **120 s** |
| **Residual model** | priced | priced | **gated** |
| **Effective compute** | `C = F + λR` | `C = F + λR` | **`C = F`** |

Unchanged in every round: the score shape `s_m = MSE_final,m × max(0.1, C_m / B_m)`,
He-initialized weights with no biases, `N(0, I)` inputs, and a 100-MLP graded suite.

### The budget has always bought ~65,000 forward passes

The Phase 2 budget is **8.08×** Phase 1's, and the MLP is 8× more expensive to run,
so the ratio between them barely moves. Every round has been calibrated to the
same ratio:

| round | budget ÷ one forward pass |
|---|---|
| `v1-warmup` | 64,850 |
| `v1-phase1` | 64,850 |
| `v2-phase2` | 65,536 |

So the shape is the variable, not the budget. Brute-force Monte Carlo has been held
at roughly the same 65k samples per MLP in every round, and in every round that is
far short of what a competitive score needs. A larger budget has never made
brute-force sampling viable.

What changed at Phase 2 is where the cost sits. At 256 × 32 most of the work comes
from depth; at 1024 × 16 most of it comes from width, and anything with an
`O(width³)` term gets 64× more expensive while an `O(width²)` term gets only 16×
more. That is why
[`examples/03_covariance_propagation.py`](../../examples/03_covariance_propagation.py)
costs **597×** what [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py)
costs at this shape, against 137.6× at the Phase 1 shape.

## What changed, round to round

**warm-up → Phase 1.** MLPs got deeper (8 → 32 layers) and the budget grew 4×. Same
rulebook otherwise.

**Phase 1 → Phase 2.** Three things moved at once:

1. **Shape: 256 × 32 → 1024 × 16.** Wider and shallower. An estimator that hardcoded
   the Phase 1 shape now returns a wrong-shaped array; one that reads `mlp.width` /
   `mlp.depth` still runs, but both its cost and its score move.
2. **Residual wall time stopped being priced.** See
   [Residual wall time: priced, then gated](#residual-wall-time-priced-then-gated).
3. **Memory dropped.** Solution-process memory went from 64 GB to **8 GB**. This is
   the one Phase 2 change that tightens a limit rather than rescaling it, and the one
   most likely to break a working Phase 1 estimator that materialized large
   intermediates.

## Residual wall time: priced, then gated

"Residual" is the part of `predict()` that flopscope does not meter: your Python,
control flow, GC. Unconstrained, it would allow work the FLOP budget does not measure,
so every round has constrained it. The mechanism changed at Phase 2.

**Priced (`v1-warmup`, `v1-phase1`).** Residual seconds were converted to FLOPs at
λ = 1e11 and added to the metered total, so `C = F + λR`. Wall time was allowed but
consumed budget, so spending more of it left fewer FLOPs.

**Gated (`v2-phase2`, current).** λ = 0, so residual pricing is **deprecated**.
Residual time is capped separately at 0.4 s, and exceeding the cap fails the MLP.
Effective compute is then `C = F`.

Under Phase 1 a slow estimator cost budget. Under Phase 2 it fails the MLP.
Residual time covers plumbing: unpacking `mlp`, control flow, and assembling the
result. See [Allowed Code](../concepts/allowed-code.md).

### Why Phase 2 does not price residual time

Phase 1 charged residual seconds at λ = 1e11 FLOPs per second. Phase 2 sets λ = 0
and caps residual time instead. Three properties of the priced model motivated the
change.

**A rate makes unmetered compute permitted rather than prohibited.** Under λ,
arithmetic performed outside flopscope had a defined cost, so a submission that spent
the corresponding budget stayed within the rules. Phase 2
[prohibits](../concepts/allowed-code.md#what-is-prohibited) computation outside
flopscope, so there is nothing for a rate to convert. The remaining use of
residual time is plumbing, which a cap bounds directly.

**The priced score depended on the grading machine.** `R_m` is wall-clock time, so
under λ > 0 an identical submission produced different values of `C_m` on different
hardware and under different load. `F_m` is derived analytically from tensor shapes
and dtypes and does not vary by machine. With λ = 0 the ranked quantity is `F_m`
alone, which does not depend on where the run happened. See
[Is scoring hardware-dependent?](../troubleshooting/faq.md#is-scoring-hardware-dependent).

**A combined budget did not state a FLOP limit.** Under λ, `B` bounded a sum of
FLOPs and priced seconds, so the FLOPs available to a submission depended on how
much residual time it spent. With λ = 0, `B` is a FLOP limit.

The 0.4 s cap is a limit, not an allowance. Residual time below it does not affect
the score: 3 ms and 300 ms both contribute nothing to `C_m`. Exceeding it zeroes
that MLP's predictions.

> The rate is deprecated, not removed. `PHASE1_LAMBDA_FLOPS_PER_SECOND` is still
> exported so a `v1-*` round can be re-scored correctly.

## Reproducing a score from an earlier round

Restore **all** of a round's settings. A partial restore scores the run under a mix
of two rulebooks and produces a number that matches neither.

```bash
uv run whest run --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
    --flop-budget 272000000000 \
    --lambda-flops-per-second 1e11 \
    --no-residual-wall-time-limit \
    --wall-time-limit 60
```

The last two flags matter as much as the budget. A submission that took between 60 s
and 120 s was `time_exhausted` under Phase 1 but passes under today's default. Phase 1
also gated nothing, so leaving today's 0.4 s residual cap in place fails MLPs that
round would have allowed.

## Why your old local numbers do not compare

A score you recorded before Phase 2 is not comparable to one you record now, for three
independent reasons:

- **The shape changed**, so `final_layer_mse` is measured over a different array
  against different ground truth.
- **The cost model changed**, so `C_m` is computed differently even for identical code.
- **The meter changed.** flopscope 0.12.0 reprices ~40 operations relative to 0.11.0,
  in both directions. See the [FAQ on meter changes](../troubleshooting/faq.md).

Re-run against `@v2-phase2` and re-measure rather than trusting a number you wrote
down earlier.

## Where these numbers come from

Every value in the side-by-side table **except the two forward-pass rows** is read from
`whestbench.budget.ROUNDS`, the single source of truth. whestbench derives its own
defaults from `CURRENT_ROUND` rather than restating them, so the harness and this page
cannot disagree:

```python
from whestbench.budget import ROUNDS, CURRENT_ROUND

CURRENT_ROUND.tag                    # 'v2-phase2'

r = ROUNDS["v1-phase1"]
r.flop_budget                        # 272_000_000_000
r.lambda_flops_per_second            # 1e11   -- priced
r.residual_wall_time_limit_s         # None   -- that round gated nothing
r.wall_time_limit_s                  # 60.0   -- not today's 120.0
r.width, r.depth                     # (256, 32)
```

The two exceptions are derived rather than read, because `RoundConfig` carries no
forward-pass field. **One forward pass** is the arithmetic count `depth · 2 · width²`,
and **Budget ÷ forward pass** divides by it. flopscope's measured cost is slightly higher:
the kit's own sampler measures **33,637,376 FLOPs** per pass at 1024×16, giving **65,374**
passes per budget. Use the arithmetic figure to compare rounds, since it depends only on
the shape; use [Local Engine API](local-engine-api.md) when you need the metered number.
The ~65,000 invariant holds either way. Measured, the three rounds come out at 64,083 /
64,281 / 65,374.

`tests/test_rounds_doc.py` asserts this page against that API on every CI run, so a
round change upstream fails the kit's build instead of leaving the table stale.

The two evaluator-side limits (the 8 GB memory cap and the 5 s `setup()` timeout)
live in `whestbench.scoring.ContestSpec` rather than in `RoundConfig`, and earlier
rounds' values for them are recorded in the [CHANGELOG](../../CHANGELOG.md) rather
than in the API.

## ➡️ Next steps

- [Scoring Model](../concepts/scoring-model.md) — how `s_m` is actually computed.
- [Manage FLOP Budget](../how-to/manage-flop-budget.md) — where your FLOPs go at this shape.
- [Use Evaluation Datasets](../how-to/use-evaluation-datasets.md) — pinning a revision.

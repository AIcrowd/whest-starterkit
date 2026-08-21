# Estimator Contract

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page when you need exact estimator I/O requirements.

## Phase 2 limits

| Limit | Value |
|---|---|
| MLP shape | `width = 1024`, `depth = 16` — so every `predict()` returns `(16, 1024)` |
| FLOP budget per MLP (`B_m`) | `2**41` = `2,199,023,255,552` |
| Wall-clock time per MLP | 120 s |
| Residual wall time per MLP | 400 ms — a hard cap, not a price |
| `setup()` | 5 s total, run once per submission |
| Solution process memory | 8 GB |

Crossing a per-MLP limit zeroes that MLP's predictions and forfeits its
compute discount — see [Failure semantics](#failure-semantics). Crossing the
`setup()` ceiling fails the whole submission.

> **Residual wall time is for plumbing, not for computing.** It is no longer
> priced into your score: it is capped at 400 ms per MLP, and an MLP that
> crosses the cap falls back to zero predictions. The cap exists so that the
> FLOP budget stays the only currency. Doing meaningful numerical work in
> residual time — uninstrumented Python arithmetic, or anything that computes
> while flopscope is not counting — is a rules violation and grounds for
> disqualification, not a trade you can pay for; see
> [Allowed Code](../concepts/allowed-code.md). If your estimator genuinely
> needs more than 400 ms of residual time for *plumbing*, write to
> <arc-whestbench@aicrowd.com> before you submit.

## Required interface

`predict(self, mlp: MLP, budget: int) -> fnp.ndarray`

Optional lifecycle hooks:

- `setup(self, context: SetupContext) -> None`
- `teardown(self) -> None`

### Lifecycle

```
  Estimator()           ──▶  __init__         (cheap; no I/O, no compute)
       │
       ▼
  setup(context)        ──▶  one call per submission, before any predict
       │                     • runs OUTSIDE any BudgetContext (off-budget)
       │                     • hard 5 s ceiling; overrunning it fails the
       │                       whole submission
       │                     • good for: loading shipped weights, lookup
       │                                  tables, config loads
       ▼
  predict(mlp_1, b)     ──▶  one call per MLP
  predict(mlp_2, b)            • runs INSIDE a BudgetContext
  ...                          • bounded by the per-MLP FLOP budget, the
  predict(mlp_M, b)              120 s wall cap and the 400 ms residual cap
       │
       ▼
  teardown()            ──▶  one call after all predict() calls
                             • cleanup of resources opened in setup()
```

`setup()` and `teardown()` are entirely optional — `examples/02_*` and
`examples/03_*` skip both. Define them when you have shape-agnostic work to
load once per run.

Two things about `setup()` on the grader are easy to get wrong:

- **It is off the FLOP budget, and off the residual too.** Nothing `setup()`
  does is billed to any MLP — not its FLOPs, not its wall time. This is real
  and you can rely on it.
- **It runs once, and 5 s is the entire allowance.** `setup()` is called a
  single time per submission, before any `predict()`, and it has a hard 5 s
  ceiling. Exceeding it fails the whole submission with `SETUP_TIMEOUT` — not
  one MLP, the whole run. `predict()` gets its own separate 120 s per MLP;
  setup cannot borrow from it.

The practical upshot: `setup()` is for *loading* precomputed work, not for
doing it. Compute offline, ship the artifact, read it here — see
[Ship Weights](../how-to/ship-weights.md). See also
[FAQ: Can I precompute things in setup()?](../troubleshooting/faq.md#can-i-precompute-things-in-setup)

### `SetupContext` fields

| Field | Type | Description |
|---|---|---|
| `width` | `int` | Neuron count for generated MLPs (`1024` in Phase 2) |
| `depth` | `int` | Number of layers per MLP (`16` in Phase 2) |
| `flop_budget` | `int` | FLOP cap for the estimator (`2**41` in Phase 2) |
| `api_version` | `str` | Contract version string |
| `scratch_dir` | `str \| None` | Optional writable directory for caching across calls (subprocess runner; otherwise typically `None`) |
| `submission_dir` | `str \| None` | Folder your submission was extracted into — locally, your estimator's folder; populated by `whest validate` / `whest run` and on the grader. Load shipped files (e.g. `weights.npz`) from here. See [how-to/ship-weights.md](../how-to/ship-weights.md). |
| `seed` | `int` | Run-level seed for setup-time randomness; the same value for every MLP in the run, `0` when no seed was configured. See [Reproducibility under the grader seed](#reproducibility-under-the-grader-seed). |

## Input object quick reference

| Object | Field | Meaning |
|---|---|---|
| `MLP` | `width` | Number of neurons per layer (`1024` in Phase 2) |
| `MLP` | `depth` | Number of weight matrices (layers) (`16` in Phase 2) |
| `MLP` | `weights` | Ordered weight matrices, each `(width, width)` |
| `MLP` | `seed` | Per-MLP grader-supplied seed for predict-time randomness. See [Reproducibility under the grader seed](#reproducibility-under-the-grader-seed). |

For traversal examples, see [Inspect and Traverse MLP Structure](../how-to/inspect-mlp-structure.md).

## Output requirements per `predict` call

| Requirement | Rule |
|---|---|
| Shape | Return a 2D array with shape `(mlp.depth, mlp.width)` — `(16, 1024)` in Phase 2 |
| Numeric validity | Every value is finite |

Read the shape off the `mlp` you were handed rather than hard-coding
`(16, 1024)`: `whest validate` probes your estimator on a tiny
`width=4, depth=2` MLP, and a hard-coded shape fails that probe.

## FLOP tracking

Your estimator must use flopscope primitives (`import flopscope as flops` and `import flopscope.numpy as fnp`) for all numerical computation. flopscope tracks FLOP usage analytically. If the total FLOPs across your entire `predict` call exceed `flop_budget`, all predictions for that MLP are replaced with zero vectors and your MSE for that MLP is computed against zeros.

## Allowed code

Your submission may use exactly three things: the grader's Python
interpreter, the flopscope client API (plus the `whestbench` contract types
this page describes), and the pure-Python standard library. Data files
shipped alongside `estimator.py` — weights, lookup tables, other precomputed
artifacts — remain permitted; see
[Ship Weights](../how-to/ship-weights.md).

Prohibited, and grounds for disqualification rather than for a higher bill:
vendored numpy/scipy/BLAS, compiled kernels of any kind, ctypes/cffi/FFI,
asyncio/threads/subprocess/multiprocessing, computing while a flopscope op is
in flight, and touching the flopscope client, transport, or accounting.

[Allowed Code](../concepts/allowed-code.md) is the full rule — what each
prohibition covers, how it is enforced, and where to send a FLOP-mispricing
report.

## Failure semantics

The harness never crashes on a bad estimator. Every failure mode is
surfaced as report data so that one bad MLP doesn't take down the run.

| Failure | Behavior | Report field(s) surfacing it | Stage that catches it first |
|---|---|---|---|
| Wrong return shape (not `(mlp.depth, mlp.width)`) | predictions for this MLP zeroed | `per_mlp[i].error.details.{expected_shape, got_shape}` | Stage 2 (`whest validate`) |
| Wrong dtype (not a `flopscope.numpy.ndarray`) | predictions for this MLP zeroed | `per_mlp[i].error` with hint | Stage 2 |
| Non-finite values (NaN, Inf) | predictions for this MLP zeroed | `per_mlp[i].error.details.cause_hints` | Stage 2 |
| `predict()` raised an exception | predictions for this MLP zeroed; harness continues to the next MLP; CLI exits `1` and prints an "Estimator Errors" panel | `per_mlp[i].{error, error_code, traceback}`; `error_code` is the Python exception class name | Stage 3 (`whest run`) |
| Exceeded `flop_budget` | flopscope raises `BudgetExhaustedError` *before* the over-budget op runs; predictions zeroed | `per_mlp[i].budget_exhausted: true` | Stage 3 |
| Exceeded the wall-clock cap (`wall_time_limit_s`; 120 s on the grader) | flopscope raises `TimeExhaustedError`; predictions zeroed | `per_mlp[i].time_exhausted: true` | Stage 3 (with `--wall-time-limit`) |
| Exceeded the residual cap (`residual_wall_time_limit_s`; 400 ms on the grader) | scoring layer (not flopscope) zeroes the predictions after `predict()` returns | `per_mlp[i].residual_wall_time_exhausted: true` | Stage 3 (with `--residual-wall-time-limit`) |

The local CLI does not default to the grader's numbers. To rehearse under
Phase 2 conditions, pass all three explicitly:

```bash
uv run whest run --estimator estimator.py --runner local \
  --flop-budget 2199023255552 --wall-time-limit 120 --residual-wall-time-limit 0.4
```

When `predict()` raises, the runner captures the exception, records the
class name in `error_code`, and forwards a formatted `traceback` (subprocess
runs forward it across the worker boundary). Use `--debug` to see
tracebacks inline; `--fail-fast` to halt at the first failure.

Predictions for the failed MLP are scored against zeros, and the compute
multiplier is forced to 1.0 (no discount), so the failure *does* hurt your
`adjusted_final_layer_score`. If you want the run to stop at the first problem
rather than score-against-zeros, use `--fail-fast`.

For the structured `error.details` schema, see
[score-report-fields.md](score-report-fields.md#per-mlp-fields).

## Reproducibility under the grader seed

The grader supplies two independent seeds. An estimator that uses randomness
must seed off them, so that a regrade reproduces the score you were ranked
on:

| Seed | Where you read it | Scope |
|---|---|---|
| `ctx.seed` | `SetupContext.seed`, inside `setup()` | one value per run, shared by every MLP |
| `mlp.seed` | `MLP.seed`, inside `predict()` | one value per MLP |

Always consume them through an isolated generator —
`fnp.random.default_rng(mlp.seed)` — never `fnp.random.seed(...)`, which
mutates process-global state. For the idioms, including spawning independent
sub-streams from one seed, see
[code-patterns.md](code-patterns.md#seed-randomness-from-mlpseed-and-ctxseed).

Seeding off a participant-chosen constant (`default_rng(42)`) instead of the
grader seeds may cost you prize eligibility.

## ➡️ Next step

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)

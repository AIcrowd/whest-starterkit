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
| `setup()` | 5 s per run — runs once per worker process, several times per submission |
| Solution process memory | 8 GB |

Crossing a per-MLP limit zeroes that MLP's predictions and forfeits its
compute discount; see [Failure semantics](#failure-semantics). Crossing the
`setup()` ceiling fails the whole submission.

These are the current round's numbers. For the earlier rounds' shapes, budgets,
λ rates and wall caps, and for the flags that reproduce a score from one, see
[Competition Rounds](rounds.md).

> **Residual wall time is for plumbing, not for computing.** It is no longer
> priced into your score: it is capped at 400 ms per MLP, and an MLP that
> crosses the cap falls back to zero predictions. The cap exists so that the
> FLOP budget stays the only currency. Doing meaningful numerical work in
> residual time (uninstrumented Python arithmetic, or anything that computes
> while flopscope is not counting) is a rules violation and grounds for
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
  setup(context)        ──▶  one call per worker process, before its predicts
       │                     • runs OUTSIDE any BudgetContext (off-budget)
       │                     • hard 5 s ceiling per run; overrunning it fails
       │                       the whole submission
       │                     • good for: loading shipped weights, lookup
       │                                  tables, config loads
       ▼
  predict(mlp_1, b)     ──▶  one call per MLP
  predict(mlp_2, b)            • runs INSIDE a BudgetContext
  ...                          • bounded by the per-MLP FLOP budget, the
  predict(mlp_M, b)              120 s wall cap and the 400 ms residual cap
       │
       ▼
  teardown()            ──▶  one call per worker, after its predicts
                             • cleanup of resources opened in setup()
                             • skipped entirely if the worker died
```

`setup()` and `teardown()` are entirely optional. The root
[`estimator.py`](../../estimator.py) defines neither and scores fine. All four
`examples/` define `setup()`: 01-03 only to carry the `ctx.seed` scaffold,
[`examples/04_shipped_weights.py`](../../examples/04_shipped_weights.py) to load
its `.npz`. Only [`examples/01_random.py`](../../examples/01_random.py) also
defines `teardown()`. Define `setup()` when you have shape-agnostic work to load
once per worker.

Two things about `setup()` on the grader are easy to get wrong:

- **It is off the FLOP budget, and off the residual too.** Nothing `setup()`
  does is billed to any MLP: not its FLOPs, not its wall time. This is real
  and you can rely on it.
- **It runs once per worker, not once per submission, and each run gets its own
  5 s.** whestbench 0.16.0 replaces a subprocess worker that dies mid-suite and
  carries on, and the replacement re-runs your `setup()`; the grader also serves
  one submission from several worker processes. Upstream puts the practical
  figure at roughly 5-15 runs per submission. Make `setup()` idempotent, and
  cost it as something you pay several times over. Exceeding the 5 s on any
  single run fails the whole submission with `SETUP_TIMEOUT`, not one MLP.
  `predict()` gets its own separate 120 s per MLP; setup cannot borrow from it.

So `setup()` is for *loading* precomputed work, not for doing it. Compute
offline, ship the artifact, read it here. See
[Ship Weights](../how-to/ship-weights.md) and
[FAQ: Can I precompute things in setup()?](../troubleshooting/faq.md#can-i-precompute-things-in-setup)

### `SetupContext` fields

| Field | Type | Description |
|---|---|---|
| `width` | `int` | Neuron count for generated MLPs (`1024` in Phase 2) |
| `depth` | `int` | Number of layers per MLP (`16` in Phase 2) |
| `flop_budget` | `int` | FLOP cap for the estimator (`2**41` in Phase 2) |
| `api_version` | `str` | Contract version string |
| `scratch_dir` | `str \| None` | Reserved. `None` on every whestbench 0.16.0 path — `whest validate`, and `whest run` under either runner. Nothing populates it, so do not plan caching around it; guard with `if ctx.scratch_dir:` if you touch it at all. Load shipped files from `submission_dir` instead. |
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

The harness does not type- or dtype-check the return value: anything with
`.shape` and `__array__` is coerced with `fnp.asarray(..., dtype=fnp.float32)`,
so a plain `numpy` array of the right shape scores without complaint. Computing
with `fnp` throughout is a rule of the round (see [Allowed code](#allowed-code)),
not something the output check enforces.

## FLOP tracking

Your estimator must use flopscope primitives (`import flopscope as flops` and `import flopscope.numpy as fnp`) for all numerical computation. flopscope tracks FLOP usage analytically. If the total FLOPs across your entire `predict` call exceed `flop_budget`, all predictions for that MLP are replaced with zero vectors and your MSE for that MLP is computed against zeros.

## Allowed code

Your submission may use exactly three things: the grader's Python
interpreter, the flopscope client API (plus the `whestbench` contract types
this page describes), and the pure-Python standard library. Data files
shipped alongside `estimator.py` (weights, lookup tables, other precomputed
artifacts) remain permitted; see
[Ship Weights](../how-to/ship-weights.md).

Prohibited, and grounds for disqualification rather than for a higher bill:
vendored numpy/scipy/BLAS, compiled kernels of any kind, ctypes/cffi/FFI,
asyncio/threads/subprocess/multiprocessing, computing while a flopscope op is
in flight, and touching the flopscope client, transport, or accounting.

[Allowed Code](../concepts/allowed-code.md) is the full rule: what each
prohibition covers, how it is enforced, and where to send a FLOP-mispricing
report.

## Failure semantics

The harness never crashes on a bad estimator. Every failure mode is
surfaced as report data so that one bad MLP doesn't take down the run.

| Failure | Behavior | Report field(s) surfacing it | Stage that catches it first |
|---|---|---|---|
| Wrong return shape (not `(mlp.depth, mlp.width)`) | predictions for this MLP zeroed | `per_mlp[i].error.details.{expected_shape, got_shape}` | Stage 2 (`whest validate`) |
| Return value has no `.shape` (a Python list, a scalar) | treated as shape `()` and zeroed | `per_mlp[i].error.details.got_shape` is `[]` | Stage 2 |
| Non-finite values (NaN, Inf) | predictions for this MLP zeroed | `per_mlp[i].error.details.cause_hints` | Stage 2 |
| `predict()` raised an exception | predictions for this MLP zeroed; harness continues to the next MLP; CLI exits `1` and prints an "Estimator Errors" panel | `per_mlp[i].{error, error_code, traceback}`. `error_code` is a harness code, not the exception class: `PREDICT_ERROR` for anything your `predict()` raises, and `WORKER_EOF` / `WORKER_BROKEN_PIPE` / `WORKER_IO_ERROR` / `PREDICT_TIMEOUT` when the subprocess worker dies. The exception class name reaches `error_code` only for harness-raised errors under `--runner local` (e.g. a shape `ValueError`); the same failure reports `PREDICT_ERROR` under `--runner subprocess`. The class name is always in `traceback`. | Stage 3 (`whest run`) |
| Exceeded `flop_budget` | flopscope raises `BudgetExhaustedError` *before* the over-budget op runs; predictions zeroed | `per_mlp[i].budget_exhausted: true` | Stage 3 |
| Exceeded the wall-clock cap (`wall_time_limit_s`; 120 s on the grader) | flopscope raises `TimeExhaustedError`; predictions zeroed | `per_mlp[i].time_exhausted: true` | Stage 3 (`whest run`, on by default) |
| Exceeded the residual cap (`residual_wall_time_limit_s`; 400 ms on the grader) | scoring layer (not flopscope) zeroes the predictions after `predict()` returns | `per_mlp[i].residual_wall_time_exhausted: true` | Stage 3 (`whest run`, on by default) |

Both caps are on by default at the graded values, and so are the other two:
since whestbench 0.16.0 the local CLI defaults to `--flop-budget 2199023255552`,
`--wall-time-limit 120`, `--setup-timeout 5` and
`--residual-wall-time-limit 0.4`. A plain

```bash
uv run whest run --estimator estimator.py --runner local
```

is already a faithful rehearsal of the limits; confirm with `--format json` and
read `run_config`. The flags exist to *change* them.
`--no-residual-wall-time-limit` turns the residual gate off entirely, which you
need only to re-score a [Phase 1 round](rounds.md).

What still differs locally is the *ground truth*, not the limits: a dataset-less
run draws `--n-samples 200_000` Monte-Carlo samples per MLP, where the Phase 2
dataset was baked at 1e9. Use `--dataset` when you want the graded targets; see
[Use Evaluation Datasets](../how-to/use-evaluation-datasets.md).

When `predict()` raises, the runner captures the exception, records a harness
code (`PREDICT_ERROR`) in `error_code`, and forwards a formatted `traceback`
(subprocess runs forward it across the worker boundary). Use `--debug` to see
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

Always consume them through an isolated generator
(`fnp.random.default_rng(mlp.seed)`), never `fnp.random.seed(...)`, which
mutates process-global state. For the idioms, including spawning independent
sub-streams from one seed, see
[code-patterns.md](code-patterns.md#seed-randomness-from-mlpseed-and-ctxseed).

Seeding off a participant-chosen constant (`default_rng(42)`) instead of the
grader seeds may cost you prize eligibility.

## ➡️ Next step

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)

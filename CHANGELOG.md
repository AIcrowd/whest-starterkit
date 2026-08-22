# Changelog

Participant-visible changes to the starter kit. The kit is the published spec of the
live round, so anything that moves a rule, a parameter, a FLOP figure, or the estimator
contract is announced here.

The kit is not published to PyPI and has never carried a version tag, so entries are
dated and grouped by the dependency baseline they shipped against. **Unreleased** is
what has landed on `main` since the last dated entry. Read it before you tune anything.

## Unreleased

### BREAKING CHANGE

- **Phase 2 is the live round.** The graded shape is now **width 1024 x depth 16**
  (Phase 1 was 256 x 32), so `predict()` returns a `(16, 1024)` array. The per-MLP FLOP
  budget `B_m` is **`2**41` = 2,199,023,255,552 FLOPs**, 8.085x the Phase 1 budget of
  ~2.72e11. Effective compute is now **`C_m = F_m`**: residual wall time is no longer
  priced and the residual-penalty rate λ does not apply in Phase 2. A Phase 1 estimator
  that reads `mlp.width` / `mlp.depth` still runs, but its cost and its score both move;
  anything that hardcoded the Phase 1 shape or budget now returns the wrong array.
- **Residual wall time is capped, not billed.** A hard cap of **400 ms per MLP** replaces
  the λ pricing; exceeding it substitutes zero predictions for that MLP. The residual
  bucket is for plumbing, not for computation. Meaningful compute there is
  disqualifiable, not a cheap trade. See
  [docs/concepts/allowed-code.md](docs/concepts/allowed-code.md).
- **Phase 2 limits.** Wall-clock cap **120 s per MLP** (was 60 s). Solution process memory
  **8 GB** (the Phase 1 box was 64 GB). `setup()` is capped at **5 s** and runs once.
  Exceeding it fails the **whole submission**, not one MLP. **10 submissions per team per
  UTC day** (was 50). Unchanged from Phase 1: 2 vCPUs pinned to the solution with 14 to
  the flopscope backend, a 100-MLP test suite (50 public + 50 holdout), and the score
  shape `s_m = MSE_final,m * max(0.1, C_m / B_m)`.
- **Allowed code is now an explicit, enforced rule.** A submission may use the grader's
  Python interpreter, the flopscope client API, and the pure-Python standard library —
  nothing else. Vendored numpy/scipy/BLAS, compiled kernels of any kind, ctypes/cffi/FFI,
  asyncio/threads/subprocess/multiprocessing, compute while a flopscope op is in flight,
  and touching the flopscope client/transport/accounting are prohibited. Data files
  (weights, lookup tables, precomputed artifacts) remain permitted. Enforcement may be
  retrospective and LLM-assisted, so a submission can be invalidated after it has been
  graded.

### Feat

- **docs**: add [Allowed Code](docs/concepts/allowed-code.md) — what a submission may use,
  the prohibition list, the data-file carve-out, what residual time is for, and how the
  rule is enforced.
- **docs**: add [Competition Rounds](docs/reference/rounds.md) — every round the challenge
  has run, side by side, with the flags needed to reproduce a score from an earlier one.
- **docs**: add this changelog, so a parameter change in the kit has somewhere to be
  announced.
- **deps**: pin `whestbench>=0.16.0,<0.17.0` and `flopscope>=0.12.0,<0.13.0`. whestbench 0.16.0
  is the first release whose *defaults* are the graded Phase 2 round (1024x16, `2**41`, a
  120 s predict cap, a 5 s `setup()` cap and a gated 400 ms residual), so under 0.15.0 the kit
  documented rules the local CLI did not enforce. flopscope 0.12.0 reprices ~40 operations
  relative to 0.11.0, but none that these examples touch: `matmul`, tagged `einsum`,
  `as_symmetric`/`is_symmetric`, `fill_diagonal`, `stats.norm.*` and `astype` bill identically
  under both, so every FLOP figure in `docs/` is unchanged (`tests/test_flopscope_cost_docs.py`
  is the gate). The evaluator is still on `whestbench@v0.15.0` + `flopscope[server]==0.11.0`;
  its open PR #265 moves it to this pair.
- **harness (whestbench 0.16.0)**: the local CLI now defaults to the graded round, so local
  numbers are the grader's numbers. New/changed on `whest run`: `--setup-timeout SECONDS`
  (new; default 5.0, and breaching it fails the *whole* submission, so rehearse it),
  `--wall-time-limit` default 60.0 -> 120.0, and `--no-residual-wall-time-limit` (new; needed
  only to re-score a Phase 1 round). `whestbench.budget.ROUNDS` / `CURRENT_ROUND` expose every
  round's settings as data.
- **harness (whestbench 0.16.0), scoring-affecting**: a subprocess worker that dies mid-suite is
  now replaced and the suite continues; previously one crash failed every remaining MLP, so a
  crashy submission's suite score improves. MLP weights are float32 on every construction path.
- **local_engine**: re-default `compare_against_monte_carlo` to the real Phase 2 budget —
  `estimator_budget=2**41`, and a `5e12` sampling budget applied per sample-count row.

### Docs

- Phase 2 content pass across `docs/` and `examples/`: every cost figure re-measured at
  1024 x 16 against whestbench 0.16.0 and flopscope 0.12.0 — and identical under flopscope
  0.11.0, since 0.12.0's repricing misses every operation these examples use. The bundled
  examples now cost **86,639,616 FLOPs (0.0039% of `B_m`)** for `02_mean_propagation` and
  **51,709,240,799 FLOPs (2.3515% of `B_m`)** for `03_covariance_propagation` — a 597x
  ratio between them, up from 137.6x at the Phase 1 shape, because the covariance step is
  cubic in width.

## 2026-08-12

### Fix

- **docs(setup)**: correct the `setup()` budget contract after evaluator #203 (#35).

## 2026-08-06

### Docs

- **troubleshooting**: document the aggregate evaluation time limits, the per-MLP time
  limit, and the graded MLP count (#34).

## 2026-08-03

### Fix

- stop claiming scoring is hardware-independent — it never was (#31).
- correct the ground-truth default, the covariance symmetry idiom, the `as_symmetric`
  cost, and a demo path leak (#31).

## 2026-07-31

### Feat

- **deps**: bump to flopscope 0.10.0 + whestbench 0.14.0.

### Fix

- **examples**: re-validate covariance symmetry each layer under flopscope 0.10.0.

### Docs

- re-baseline cost numbers and meter notes for flopscope 0.10.0 / whestbench 0.14.0.

## 2026-07-24

### BREAKING CHANGE

- flopscope 0.9 moves to four-factor billing, so absolute FLOP counts change: sorts bill
  `4*N*ceil(log2 N)` rather than a flat 4x, and a reshape bills even where it returns a
  view. Every cost figure in the kit was re-derived; a Phase 1 estimator's `flops_used`
  moves without any change on your side.

### Feat

- **deps**: adopt whestbench 0.13.0 and flopscope 0.9.1.
- **local_engine**: mirror the upstream float32 sampling idiom and `MLP.seed` passthrough.

### Docs

- rewrite the flopscope primer's cost sections for four-factor billing; re-baseline the
  code-patterns cost guidance; re-profile example 02; correct the on-the-fly `n-samples`
  default to `width*width*256` and re-derive the noise floor; document `time_source=bake`
  sampling attribution in the score report; describe `whest profile-simulation` as the
  timing/correctness benchmark it is.

### Test

- drift-gate the flopscope 0.9.x billing facts the docs state.

## 2026-06-27

### Feat

- **deps**: bump whestbench to 0.12.0rc3 and the flopscope floor to 0.8.0rc5 (#29).

## 2026-06-24

### Docs

- **troubleshooting**: document common participant failure modes (#28).

## 2026-06-23

### Feat

- **deps**: pin `whestbench>=0.12.0rc2` (#25).

### Docs

- document the 4 GiB per-array memory cap (#26).
- correct the grader dependency model — there is no `requirements.txt` install step (#24).
- add the MIT LICENSE and fix the broken license link (#27).

## 2026-06-19

### Docs

- clarify `whest package` / `whest submit` ergonomics: file vs folder, credential
  exclusion, preview (#21).
- rename the README heading to "ARC WhiteBox Estimation Challenge 2026 - Starter Kit" and
  add the challenge badge hub (#23).

### Chore

- refresh `demo.cast` / `demo.gif` on 0.12.0rc0 and add the headless cast pipeline (#22).

## 2026-06-18

### Feat

- **phase 1 release**: the `v1-phase1` dataset revision, a 256 x 32 shape at a 2.72e11
  per-MLP budget, on flopscope 0.8.0rc0 / whestbench 0.11.0rc0 (#17).
- add `.whestignore` so a packaged submission stays under the 50-file cap (#18).

### Fix

- correct stale warmup-era numbers missed in the phase-1 re-profile (#19).

## 2026-06-08

### Feat

- **deps**: bump whestbench 0.10.0 / flopscope 0.5.0 and re-baseline the FLOP docs (#15).

### Docs

- document the GPU/torch flags for baking datasets (#16).

## 2026-06-04

### Docs

- align the staged ladder with the public-dataset workflow and re-record the demo against
  the public Mini split (#14).
- correct ground-truth precision and the noise floor; align scoring and defaults with
  whestbench 0.9.2 (#13).

## 2026-06-01

### Feat

- **deps**: lock whestbench 0.9.2 + flopscope 0.4.2 (#11).

### Docs

- remove the fictional Docker runner stage and renumber the ladder to five stages (#10).

## 2026-05-27

### Feat

- migrate the starters to the PyPI `whestbench` release and Hugging Face datasets (#4).

### Docs

- mini-first quickstart and modernised CLI references (#6); mention the prepared-Arrow
  fast path on Hugging Face (#7).

## 2026-04-24

### Feat

- scaffold the standalone starter kit: examples 01-04, the `local_engine.py` helpers
  (`build_mlp`, `monte_carlo_layer_means`, `compare_against_monte_carlo`), a root
  `estimator.py` template, the five-stage tutorial ladder, the reference / how-to /
  troubleshooting doc tree, and CI that runs every command the README prints.

Earlier history for the examples and docs predates this repository; it was imported
with history preserved when the kit was split out into its own repo.

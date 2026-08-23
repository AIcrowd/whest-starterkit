# Allowed code

> [← Documentation](../README.md)

## 🎯 When to use this page

Read this page before you reach for anything that is not flopscope or the standard library: a bundled wheel, a compiled kernel, a thread pool, a technique that keeps your own code running while a flopscope op is in flight. This is a rule of the round, not a style guide. Code outside it is not "priced differently"; it is disqualifiable, and the check can run long after your submission has been graded and ranked.

## 📌 TL;DR

- A submission may use exactly three things: **the grader's Python interpreter**, **the flopscope client API**, and **the pure-Python standard library**.
- Everything on the [prohibition list](#what-is-prohibited) is excluded: vendored numpy/scipy/BLAS, compiled kernels, FFI, concurrency, compute that overlaps a metered op, and anything that touches the flopscope client, transport, or accounting.
- **Data files remain permitted.** Shipping weights, lookup tables, and precomputed artifacts is explicitly allowed. See [Ship Weights](../how-to/ship-weights.md).
- **Residual wall time is for plumbing, not for computation.** It is not priced in Phase 2 (`C_m = F_m`); it is hard-capped at **400 ms per MLP**. Meaningful computation there is a breach of this rule, not a cheap trade. Pricing is exactly what Phase 1 did, and dropping it is a direct consequence of this rule ([why](../reference/rounds.md#why-phase-2-does-not-price-residual-time)).
- **Every submission is reviewed.** Automated checks run first, flagged submissions go through agent-assisted validation, and a person reviews anything still unresolved. Review continues after grading, and a submission that does not conform is invalidated once identified.

## Why there is a rule at all

The challenge measures one thing: how accurately you can predict per-neuron means, and how little compute you spend doing it. That measurement only means something if every FLOP a submission spends is visible to the meter. flopscope counts array work analytically, so two submissions running the same algorithm bill the same amount no matter whose machine they land on. Arithmetic that reaches the CPU by some other path (a bundled BLAS, a compiled kernel, a thread you spawned) does the same work and is charged nothing for it. That is not a clever optimization inside the rules; it is arithmetic that leaves the measurement, and it makes your score incomparable with everyone else's.

So the allowed surface is deliberately small, and the boundary is drawn at "can the meter see it", not at "is it fast".

## What you may use

**The grader's Python interpreter.** Your estimator runs in the CPython process the grader starts. You get its bytecode, its builtins, and its garbage collector. You do not get anything you compile, link, or launch yourself.

**The flopscope client API.** `import flopscope as flops` and `import flopscope.numpy as fnp` are the only array path. Every op you call through it is billed analytically against `B_m`, which is why it is the allowed one. (The `whestbench` contract types — `BaseEstimator`, `MLP`, `SetupContext` — are the harness the grader calls you through, not a computation library; importing them is expected. See [Estimator Contract](../reference/estimator-contract.md).)

**The pure-Python standard library.** `math`, `itertools`, `collections`, `json`, `pathlib`, and the rest of the pure-Python standard library are available for control flow, bookkeeping, and loading your own shipped files. Two caveats. First, "in the standard library" does not override the prohibition list below: `ctypes`, `threading`, `multiprocessing`, `subprocess`, `asyncio`, and `concurrent.futures` are stdlib modules and are still prohibited. Second, a pure-Python loop over neurons is legal but unmetered, so it burns [residual wall time](#residual-time-is-plumbing-not-a-second-budget) rather than FLOPs, and residual time is what the 400 ms cap governs.

## What is prohibited

A submission must not ship, import, or invoke any of the following:

- vendored numpy/scipy/BLAS
- compiled kernels of any kind
- ctypes/cffi/FFI
- asyncio/threads/subprocess/multiprocessing
- compute while a flopscope op is in flight
- touching the flopscope client/transport/accounting

The first three are the same failure in three forms: real arithmetic executed outside the meter, billed at zero. The fourth adds a second execution context the meter does not model, so work disappears from the FLOP count and is not reflected in the residual wall-time measurement either. The fifth is the same idea without a thread: arranging for your own computation to proceed alongside a metered op, through a callback, a hook, or a lazily-evaluated object, so that it runs uncounted. The sixth covers patching the client, intercepting the transport, and editing the accounting after the fact: the meter is not part of your submission, and altering it invalidates the measurement it produces.

None of these are things the grader merely charges you for. There is no price at which they become allowed.

## Data files remain permitted

Precomputation is a legitimate strategy and stays legitimate in Phase 2. You may ship data files (trained weights, lookup tables, calibration constants, any precomputed artifact) alongside your estimator, and load them in `setup()`. The distinction is *data versus execution*: a `.npz` of numbers you computed offline is data, and loading it is allowed; a compiled `.so`, a vendored wheel, or a bundled interpreter extension is execution, and is not.

Two practical notes carry over from [Ship Weights](../how-to/ship-weights.md): flopscope loads **pickle-free** arrays only, and a multi-file submission must be packaged as a folder (`whest package --estimator .`) or your data file will not ship.

## Residual time is plumbing, not a second budget

Phase 2 does not price residual wall time. Effective compute is `C_m = F_m`, the FLOPs flopscope counted. Instead, residual wall time is bounded directly: a **hard cap of 400 ms per MLP**, and an MLP that exceeds it is scored against zero predictions.

The cap is a plumbing allowance, not free compute. It exists so the Python that glues your metered ops together (shaping arguments, walking layers, loading a table) has room to run. It is not an unpriced 400 ms in which to do arithmetic the meter cannot see. Doing meaningful computation there is a breach of this rule and is disqualifiable, whether or not you stay under the cap.

If your estimator genuinely needs more residual headroom than the cap allows for legitimate plumbing, ask before you submit (see [Questions](#questions)).

## How this is enforced

Automated checks run on every submission. Anything they flag goes through further
agent-assisted validation, and anything still unresolved after that is reviewed by a
person.

Review continues after grading, so a submission can be scored and listed on the
leaderboard while its review is still open. Any submission that does not conform to
these rules is invalidated once identified.

If you are unsure whether a technique falls inside the rule, ask before you submit.
See [Questions](#questions).

## Questions

Rules questions, FLOP-mispricing reports, and residual-cap exceptions go to [arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com). Include your submission ID and, for a mispricing report, a minimal snippet plus the `flops_used` you expected and the one you got. If you believe flopscope is mispricing an operation, report it; do not bypass the accounting.

## ➡️ Next step

- [Scoring Model](./scoring-model.md) — where `C_m`, `B_m`, and the multiplier come from.
- [Ship Weights](../how-to/ship-weights.md) — the allowed way to bring precomputed work with you.
- [Pre-Submission Checklist](../how-to/pre-submission-checklist.md) — the last pass before you submit.

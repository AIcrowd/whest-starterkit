# Pre-Submission Checklist

> [← Documentation](../README.md)

## 🎯 When to use this page

The minute before you click "submit" on AIcrowd. Run through these
checks; each one maps to a single command or a one-line confirmation.

> **You get 10 submissions per team per UTC day.** The counter resets at
> 00:00 UTC and unused slots do not roll over, so every submission that fails
> on something mechanical costs you a tenth of the day. Every check below
> costs less than the slot it saves.

## Correctness

- [ ] **`uv run whest validate --estimator estimator.py`** ends with a
      green `Status: success` panel. (Catches: wrong shape, non-finite
      values, broken `setup()`.)
- [ ] **`uv run whest run --estimator estimator.py --runner local --seed 42 --n-mlps 3`**
      produces an `adjusted_final_layer_score` you recognize.
- [ ] **`uv run whest run --estimator estimator.py --runner subprocess --seed 42 --n-mlps 3`**
      produces a score within ~1% of the local-runner score above.
      (Catches: shared global state, RNG re-seed differences, imports
      that fail in clean processes — see [FAQ](../troubleshooting/faq.md#my-local-score-is-great-but-my-submission-scores-10x-worse--why).)

## Budget hygiene

- [ ] In the run report, **`per_mlp[i].budget_exhausted` is `false`** for
      every MLP. Any `true` means that MLP scored against zeros.
- [ ] **`flops_used`** is comfortably under `flop_budget` — leaves headroom
      for the harder MLPs in the grader suite. The per-MLP budget is
      `B_m = 2,199,023,255,552` FLOPs (2^41), and what it caps is
      `C_m = F_m`: your analytical FLOP count, nothing else.
- [ ] **`residual_wall_time_s` stays under 0.4 s on every MLP.** Residual
      wall time is not priced into your score — it is capped: spend more than
      **400 ms** outside flopscope ops on one MLP and that MLP's prediction
      is replaced with zeros. The residual exists for plumbing — loops,
      control flow, bookkeeping — not for computation, and doing meaningful
      compute there is disqualifiable rather than merely expensive.
      Reproduce the cap locally with `--residual-wall-time-limit 0.4` and
      confirm **`residual_wall_time_exhausted` is `false`**. If you believe
      a legitimate estimator cannot fit the cap, write to
      [arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com) rather than shipping and hoping.
- [ ] **`per_mlp[i].time_exhausted` is `false`.** The grader allows **120 s
      of wall time per MLP**; confirm the value you were scored under in
      `run_config.wall_time_limit_s` in your own run report rather than
      assuming it. Re-run with `--max-threads 1` as a stress test: on the
      grader your solution process gets **2 pinned vCPUs** (the flopscope
      backend has its own 14), and the rules guarantee no particular
      evaluation hardware — so work that finishes in time locally only
      because it is spread across your laptop's cores is a bet, not a plan.
      See
      [Is scoring hardware-dependent?](../troubleshooting/faq.md#is-scoring-hardware-dependent).
- [ ] **You fit in 8 GB.** That is the whole solution process, and width 1024
      makes it a live constraint: one full covariance matrix is 1024×1024,
      sixteen times the elements of the Phase 1 256×256. If you keep a
      per-layer covariance, count the layers before you ship.
- [ ] **No internal deadline calibrated to your own machine's clock.** If
      your estimator bails out on `time.time()` and returns a fallback, a
      slower grading machine can trigger that path silently — you get a
      valid-looking score computed from the degraded answer, with no error
      to tell you. Prefer a budget-based cutoff.

## Allowed code

Phase 2 states this as a rule, not as a limitation of the sandbox: a
submission may use the **flopscope client API** and the **pure-Python
standard library**, and nothing else. Breaching it is disqualifiable, not
merely broken, and enforcement is retrospective — a submission can be
invalidated after it has been scored. Read the list once even if your
estimator is fifty lines of `fnp`; the full rule is in
[Allowed Code](../concepts/allowed-code.md).

- [ ] **No vendored array or math libraries.** Shipping `numpy`, `scipy`, a
      BLAS build, or any part of one inside your submission folder is
      prohibited, and so is any compiled kernel — `.so`, `.dylib`, `.pyd`, a
      built extension module, generated machine code.
- [ ] **No FFI.** `ctypes`, `cffi`, and any other route out of the
      interpreter are prohibited.
- [ ] **No concurrency.** `asyncio`, `threading`, `subprocess`, and
      `multiprocessing` are prohibited — including the "just to overlap I/O"
      uses.
- [ ] **No compute while a flopscope op is in flight.** Doing your own work
      alongside an in-flight flopscope call is prohibited; it is the same
      offence as computing in the residual, and it is disqualifiable.
- [ ] **Hands off the accounting.** Do not patch, wrap, replace, or
      otherwise reach into the flopscope client, its transport, or its FLOP
      accounting.
- [ ] **Data files are still allowed.** Weights, lookup tables, and
      precomputed artifacts are explicitly permitted, and with `C_m = F_m`
      they are the cheapest thing you can ship — see
      [ship-weights.md](./ship-weights.md), which also covers where the
      data/code line sits.
- [ ] **Only sandbox-available imports.** The sandbox enforces the rule from
      the other side: your estimator can import **only** `flopscope` (incl.
      `flopscope.numpy as fnp`), the `whestbench` API (`BaseEstimator`,
      `MLP`, `SetupContext`), and the standard library — there is no
      `requirements.txt` install. Your local venv *has* the rest, so a local
      run won't flag a stray `import`; grep your estimator and route all
      array math through `flopscope.numpy as fnp`. Heavier work (a
      PyTorch-trained model, a scipy routine) goes **offline** → ship a
      pickle-free `.npz`, loaded in `setup()`.

Unsure whether a technique you have in mind is inside the line? Ask
[arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com) before you spend a week on it.

## Reproducibility

- [ ] No filesystem reads from outside `SetupContext.submission_dir` (your shipped files) and `SetupContext.scratch_dir`. The
      grader can't see your laptop.
- [ ] No network calls in `setup()` or `predict()`. The grader has no
      outbound network.
- [ ] No time-based seeds (`time.time()`, `os.urandom`, …) and no
      participant-chosen seeds. If your estimator uses randomness inside
      `predict()`, seed it from `mlp.seed`:
      `fnp.random.default_rng(mlp.seed)`. If your estimator uses randomness
      inside `setup()` (e.g. a fixed random projection basis), seed it from
      `ctx.seed`: `fnp.random.default_rng(ctx.seed)`. Custom seeds at either
      site may be disqualified for prize eligibility — see
      [Estimator Contract: Reproducibility](../reference/estimator-contract.md#reproducibility-under-the-grader-seed).
      Do **not** call `fnp.random.seed(...)` — use `default_rng(...)` for an
      isolated `Generator`.

## Sanity

- [ ] `predict()` returns the **post-ReLU** mean for **every** layer,
      shape `(mlp.depth, mlp.width)` — `(16, 1024)` at the Phase 2 shape.
      Off-by-one (returning depth+1 or depth-1 layers) is the most common
      silent bug, and a hard-coded `(32, 256)` left over from Phase 1 is the
      second.
- [ ] If you ship a `setup()`: it's idempotent and stays well under the
      grader's hard **5 s** setup budget. `setup()` runs **once**, and
      overrunning those 5 s fails the whole submission with `SETUP_TIMEOUT`
      — not one MLP, all 100. Heavy precompute belongs offline: ship the
      artifact next to your estimator and load it (see
      [how-to/ship-weights.md](./ship-weights.md)).
- [ ] No `print()` left in `predict()`. The grader runs many MLPs;
      stdout flooding is a reliable way to lose `residual_wall_time_s`.

## Final command

Once every box above is checked, ship it (run `whest login` first if you
haven't):

```bash
uv run whest submit --estimator estimator.py --watch
```

`whest submit` packages, uploads, and creates the submission in one step.
Prefer to inspect the artifact first? Build it with
`uv run whest package --estimator estimator.py --output submission.tar.gz`, check
`tar tf submission.tar.gz` (it should contain `estimator.py` and `manifest.json`),
then `uv run whest submit submission.tar.gz`.

Shipping weights or extra modules? Package the **folder** instead
(`uv run whest package --estimator . --output submission.tar.gz`) — it lists every
file and asks you to confirm, and credential files like `.env` are never
included. See [Ship Weights](./ship-weights.md).

That command spends one of your **10 slots for the UTC day**. If two of you
are iterating on the same team, agree who is spending them before you both
press submit.

## ➡️ See also

- [Stage 5: Package Your Submission](../getting-started/stage-5-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
- [Score Report Fields](../reference/score-report-fields.md)

# Pre-Submission Checklist

> [← Documentation](../README.md)

## 🎯 When to use this page

The minute before you click "submit" on AIcrowd. Run through these
checks; each one maps to a single command or a one-line confirmation.

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
- [ ] **`per_mlp[i].time_exhausted`** and
      **`residual_wall_time_exhausted`** are also `false` (only relevant if
      you set `--wall-time-limit` or `--residual-wall-time-limit`).
- [ ] **`flops_used`** is comfortably under
      `flop_budget` — leaves headroom for the harder MLPs in the grader
      suite.

## Reproducibility

- [ ] **Only sandbox-available imports.** At grading time your estimator can
      import **only** `flopscope` (incl. `flopscope.numpy as fnp`), the
      `whestbench` API (`BaseEstimator`, `MLP`, `SetupContext`), and the Python
      standard library — there is no `requirements.txt` install, so `numpy`,
      `scipy`, `torch`, … are not available. Your local venv *has* them, so a
      local run won't flag a stray `import`; grep your estimator and route all
      array math through `flopscope.numpy as fnp`. Heavier work (a
      PyTorch-trained model, a scipy routine) goes **offline** → ship a
      pickle-free `.npz`, loaded in `setup()` — see
      [ship-weights.md](./ship-weights.md).
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
      shape `(mlp.depth, mlp.width)`. Off-by-one (returning depth+1 or
      depth-1 layers) is the most common silent bug.
- [ ] If you ship a `setup()`: it's idempotent and stays under the ~5s
      `setup_timeout_s`. Heavy precompute belongs in
      `SetupContext.scratch_dir` — or precompute offline and ship the artifact
      next to your estimator (see [how-to/ship-weights.md](./ship-weights.md)).
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

## ➡️ See also

- [Stage 5: Package Your Submission](../getting-started/stage-5-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
- [Score Report Fields](../reference/score-report-fields.md)

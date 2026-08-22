# Stage 5: Package Your Submission

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · [2](stage-2-validate.md) · [3](stage-3-run-local.md) · [4](stage-4-run-subprocess.md) · **5**

You've climbed the ladder. Now ship it.

> Before you click "submit", run through the
> [Pre-Submission Checklist](../how-to/pre-submission-checklist.md). It's
> one screen, all commands, and catches the bugs the grader will hit.

## 🚀 Package it

Most submissions are a single, self-contained `estimator.py`. Package that file:

```bash
uv run whest package --estimator estimator.py --output submission.tar.gz
```

This ships **only `estimator.py`** (plus a generated `manifest.json`). Before writing the archive, `whest` prints exactly what it's bundling.

> **`whest package --estimator <path>` — the path you give decides what ships:**
> - **A file** (`--estimator estimator.py`) ships **only that file**. This is the default, and all you need for a single-file estimator like this kit's.
> - **A folder** (`--estimator .`) ships **every file in that folder**, for embedding weights or splitting across modules (see below).
> - **If your single file imports a module beside it, packaging stops.** `whest` refuses
>   rather than shipping an archive that would `ImportError` on the grader:
>   `Error [package:PACKAGING_VALIDATION_ERROR]: estimator.py imports helpers, which lives
>   beside it and will NOT be bundled`. Either inline the helper, or switch to folder mode
>   (`--estimator .`).
>
> **Credential files never ship**, in either mode: `.env`, `.env.*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.keystore`, `*.jks`, SSH keys (`id_rsa*`, `id_dsa*`, `id_ecdsa*`, `id_ed25519*`), `.netrc`, `.pypirc`, `.npmrc`, `.aws`, `.ssh`, `.gnupg`, `credentials`, `credentials.*`. The full list is `_SECRET_IGNORES` in whestbench's `packaging.py`. Anything outside those patterns ships, so name your secrets accordingly: your submission goes to a public leaderboard.

## 📦 Embedding weights or multiple modules (power users)

Pre-computed `weights.npz`, or an estimator split across modules? Ship the whole
folder instead:

```bash
uv run whest package --estimator . --output submission.tar.gz
```

Folder mode needs a file named `estimator.py` at the root of that folder; it is the
entrypoint the grader imports. Without one you get `Folder submission requires an
estimator.py in <dir>, but none was found`, and the same applies if a
`.gitignore`/`.whestignore` pattern happens to match it.

Folder mode bundles every file in the folder (this kit's `.whestignore` already
keeps `docs/`, `tests/`, `examples/`, and local-only tooling out). Before it
writes anything, `whest`:

- **lists every file it will ship** and asks `[y/N]` to confirm (pass `--yes` to skip the prompt in CI);
- **never includes credential files** (`.env`, `*.pem`, keys, …);
- honors `.gitignore` / `.whestignore`; add patterns there to drop scratch or large artefacts. The 50 MiB / 50 file caps still apply.

For the full walkthrough, including how to compute `weights.npz` and which
`flopscope` ops you can use, see
[Ship Weights and Multi-File Submissions](../how-to/ship-weights.md)
and the [Flopscope Primer](../reference/flopscope-primer.md).

## 📤 Submit to AIcrowd

Ship it straight from the CLI; no manual portal upload needed.

First, log in once with your
[AIcrowd API key](https://www.aicrowd.com/participants/me/edit), or set
`AICROWD_API_KEY`, or pass `whest login --api-key <key>`:

```bash
uv run whest login
```

Then submit. `whest submit --estimator estimator.py` packages your single file
and uploads it in one step, showing the same preview first. (Power users
embedding weights: point `--estimator` at `.` to ship the folder.) You can also
submit a prebuilt tarball:

```bash
# package + submit your single-file estimator
uv run whest submit --estimator estimator.py

# or submit a tarball you already built
uv run whest submit submission.tar.gz
```

Add `--watch` to follow the submission until it's graded:

```bash
uv run whest submit --estimator estimator.py --watch
```

> **Ten submissions per team per UTC day.** The counter resets at 00:00 UTC,
> and a submission that fails on the grader still spends one, so rehearse at
> [Stage 4](stage-4-run-subprocess.md) before you spend a slot.

Prefer the browser? The packaged `submission.tar.gz` still uploads fine on
the AIcrowd challenge submission page.

## What's in the artifact

Single file (`--estimator estimator.py`):
- `estimator.py` — your file, byte-for-byte, **renamed to `estimator.py`** if it wasn't already (the manifest entrypoint is always the `estimator` module)
- `manifest.json` — entrypoint, whestbench/flopscope/numpy versions, Python version, per-file SHA-256, and package timestamp

Folder (`--estimator .`): every non-ignored file in the folder (helper modules,
`weights.npz`, …) plus `manifest.json`. Requires an `estimator.py` at the folder root.

> **No third-party packages on the grader.** Your estimator runs in a
> locked-down sandbox that provides only `flopscope` (incl.
> `flopscope.numpy as fnp`), the `whestbench` API (`BaseEstimator`, `MLP`,
> `SetupContext`), and the Python standard library. There is no
> `requirements.txt` install step, so `numpy`, `scipy`, `torch`, … are not
> importable. Do anything that needs them **offline** and ship the result as a
> pickle-free `.npz` (see [Ship Weights](../how-to/ship-weights.md)).

## After submission

What happens once `whest submit` (or a portal upload) accepts your
`submission.tar.gz`:

1. **AIcrowd unpacks the artifact** and runs your estimator in a locked-down
   sandbox that provides **only** `flopscope` (incl. `flopscope.numpy as fnp`),
   the `whestbench` API (`BaseEstimator`, `MLP`, `SetupContext`), and the
   Python standard library. There is **no third-party package install step**:
   a `requirements.txt` has no effect, and `numpy`/`scipy`/`torch` are not
   importable (do that work offline; see [Ship Weights](../how-to/ship-weights.md)).
2. **The grader runs your estimator** against a held-out
   MLP suite (same `width`, `depth`, `flop_budget` as the public
   defaults; same `n_mlps` order of magnitude), in an isolated
   subprocess inside a sandboxed container. No network, no GPU,
   no access to the local filesystem outside `SetupContext.submission_dir` (your shipped files) and `SetupContext.scratch_dir` (which is `None` on every local `whest` path, so guard it; see [Stage 4](stage-4-run-subprocess.md)).
3. **Your `setup()` runs once per worker, not once per submission**: roughly
   **5-15 times** in practice, since a submission is served by several worker
   processes and a worker that dies mid-suite is replaced and re-runs it. Each run
   stays off the FLOP budget and off the residual, and each gets its own hard **5 s**
   ceiling. If any one raises, or overruns that ceiling, the **whole** submission is
   recorded as failed, with the traceback surfaced in the AIcrowd UI. Keep `setup()`
   idempotent and cheap on every call (see
   [Estimator Contract: Lifecycle](../reference/estimator-contract.md#lifecycle)).
4. **`predict()` is called per MLP.** Errors per call are captured but
   don't kill the run: predictions for that MLP are scored against zeros **and that
   MLP's score multiplier is forced to 1.0** instead of the usual `max(0.1, C_m / B_m)`.
   Since a healthy estimator sits at the 0.1 floor, a failed MLP costs exactly 10x what
   the same MLP would cost if you had simply returned zeros. That is why repeated
   failures tank `adjusted_final_layer_score`. The same forcing applies when a time or
   FLOP cap trips (see [Problem Setup: computational model](../concepts/problem-setup.md#computational-model)).
5. **The leaderboard updates** with `adjusted_final_layer_score` once the run
   finishes.

If the leaderboard score disagrees with your Stage 4 score by more than
a percent or two, the suspects are listed in the
[FAQ](../troubleshooting/faq.md#my-local-score-is-great-but-my-submission-scores-10x-worse--why).

If you suspect a grader-side issue (your submission errors out without
your local Stage 4 doing so), open a thread on the
[challenge discussion forum](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026)
with the submission ID; that's the quickest path to a human.

For anything that doesn't belong in a public thread — a rules question, an
operation you believe flopscope is mispricing, or a legitimate need for more
residual wall time than the 400 ms cap allows — email
[arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com) with your submission ID and a minimal repro.

## ✅ Expected outcome

| Stage | What you should see | Action if not |
|---|---|---|
| Local Stage 4 score | ≈ leaderboard score within ~1–2% | Check Stage 4 vs Stage 3 first; drift between them surfaces the same bugs that the grader will hit |
| `submission.tar.gz` size | Typically 1–10 KB for a pure-Python estimator (this kit's stub packages to 1.7 KB); tens of MB if you ship weight files (50 MiB cap enforced by `whest package`) | If unexpectedly large, check for scratch files and use `.whestignore` to exclude them |
| Grader runtime | A few minutes for the default suite | Slower than that suggests `residual_wall_time_s` issues; see [score-report-fields.md](../reference/score-report-fields.md) |

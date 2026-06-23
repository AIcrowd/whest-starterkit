# Stage 5: Package Your Submission

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · [2](stage-2-validate.md) · [3](stage-3-run-local.md) · [4](stage-4-run-subprocess.md) · **5**

You've climbed the ladder. Now ship it.

> Before you click "submit", run through the
> [Pre-Submission Checklist](../how-to/pre-submission-checklist.md) — it's
> one screen, all commands, and catches the bugs the grader will hit.

## 🚀 Package it

Most submissions are a single, self-contained `estimator.py`. Package that file:

```bash
uv run whest package --estimator estimator.py --output submission.tar.gz
```

This ships **only `estimator.py`** (plus a generated `manifest.json`). Before writing the archive, `whest` prints exactly what it's bundling.

> **`whest package --estimator <path>` — the path you give decides what ships:**
> - **A file** (`--estimator estimator.py`) ships **only that file**. This is the default, and all you need for a single-file estimator like this kit's.
> - **A folder** (`--estimator .`) ships **every file in that folder** — for embedding weights or splitting across modules (see below).
>
> **Credential files never ship**, in either mode — `.env`, `*.pem`, `*.key`, `id_*`, `.aws/`, … are always excluded for security, because your submission goes to a public leaderboard.

## 📦 Embedding weights or multiple modules (power users)

Pre-computed `weights.npz`, or an estimator split across modules? Ship the whole
folder instead — **safely, with full visibility**:

```bash
uv run whest package --estimator . --output submission.tar.gz
```

Folder mode bundles every file in the folder (this kit's `.whestignore` already
keeps `docs/`, `tests/`, `examples/`, and local-only tooling out). Before it
writes anything, `whest`:

- **lists every file it will ship** and asks `[y/N]` to confirm (pass `--yes` to skip the prompt in CI) — nothing ships by surprise;
- **never includes credential files** (`.env`, `*.pem`, keys, …);
- honors `.gitignore` / `.whestignore` — add patterns there to drop scratch or large artefacts. The 50 MiB / 50 file caps still apply.

Full walkthrough — including how to compute `weights.npz` and which `flopscope`
ops you can use — see [Ship Weights and Multi-File Submissions](../how-to/ship-weights.md)
and the [Flopscope Primer](../reference/flopscope-primer.md).

## 📤 Submit to AIcrowd

Ship it straight from the CLI — no manual portal upload needed.

First, log in once with your AIcrowd API key (grab it from your
[AIcrowd profile](https://www.aicrowd.com/participants/me/customize)):

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

Prefer the browser? The packaged `submission.tar.gz` still uploads fine on
the AIcrowd challenge submission page.

## What's in the artifact

Single file (`--estimator estimator.py`):
- `estimator.py` — verbatim copy of yours
- `manifest.json` — entrypoint, whestbench/flopscope/numpy versions, Python version, per-file SHA-256, and package timestamp

Folder (`--estimator .`): every non-ignored file in the folder (helper modules,
`weights.npz`, …) plus `manifest.json`.

> **No third-party packages on the grader.** Your estimator runs in a
> locked-down sandbox that provides only `flopscope` (incl.
> `flopscope.numpy as fnp`), the `whestbench` API (`BaseEstimator`, `MLP`,
> `SetupContext`), and the Python standard library — there is no
> `requirements.txt` install step, so `numpy`, `scipy`, `torch`, … are not
> importable. Do anything that needs them **offline** and ship the result as a
> pickle-free `.npz` (see [Ship Weights](../how-to/ship-weights.md)).

## After submission

What happens once `whest submit` (or a portal upload) accepts your
`submission.tar.gz`:

1. **AIcrowd unpacks the artifact** and runs your estimator in a locked-down
   sandbox that provides **only** `flopscope` (incl. `flopscope.numpy as fnp`),
   the `whestbench` API (`BaseEstimator`, `MLP`, `SetupContext`), and the
   Python standard library. There is **no third-party package install step** —
   a `requirements.txt` has no effect, and `numpy`/`scipy`/`torch` are not
   importable (do that work offline; see [Ship Weights](../how-to/ship-weights.md)).
2. **The grader runs your estimator** against a held-out
   MLP suite (same `width`, `depth`, `flop_budget` as the public
   defaults; same `n_mlps` order of magnitude), in an isolated
   subprocess inside a sandboxed container. No network, no GPU,
   no access to the local filesystem outside `SetupContext.submission_dir` (your shipped files) and `SetupContext.scratch_dir`.
3. **Your `setup()` runs once.** If it raises, the run is recorded as a
   failed submission with the traceback surfaced in the AIcrowd UI.
4. **`predict()` is called per MLP.** Errors per call are captured but
   don't kill the run — predictions for that MLP are scored against
   zeros. Repeated failures will tank `adjusted_final_layer_score`.
5. **The leaderboard updates** with `adjusted_final_layer_score` once the run
   finishes.

If the leaderboard score disagrees with your Stage 4 score by more than
a percent or two, the suspects are listed in the
[FAQ](../troubleshooting/faq.md#my-local-score-is-great-but-my-submission-scores-10x-worse--why).

If you suspect a grader-side issue (your submission errors out without
your local Stage 4 doing so), open a thread on the
[challenge discussion forum](https://www.aicrowd.com/) with the
submission ID — that's the quickest path to a human.

## ✅ Expected outcome

| Stage | What you should see | Action if not |
|---|---|---|
| Local Stage 4 score | ≈ leaderboard score within ~1–2% | Check Stage 4 vs Stage 3 first — drift between them surfaces the same bugs that the grader will hit |
| `submission.tar.gz` size | Typically 2–10 KB for a pure-Python estimator; tens of MB if you ship weight files (50 MiB cap enforced by `whest package`) | If unexpectedly large, check for scratch files and use `.whestignore` to exclude them |
| Grader runtime | A few minutes for the default suite | Slower than that suggests `residual_wall_time_s` issues — see [score-report-fields.md](../reference/score-report-fields.md) |

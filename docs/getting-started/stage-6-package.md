# Stage 6: Package Your Submission

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · [2](stage-2-validate.md) · [3](stage-3-run-local.md) · [4](stage-4-run-subprocess.md) · [5](stage-5-run-docker.md) · **6**

You've climbed the ladder. Now ship it.

> Before you click "submit", run through the
> [Pre-Submission Checklist](../how-to/pre-submission-checklist.md) — it's
> one screen, all commands, and catches the bugs the grader will hit.

## 🚀 Run it

```bash
uv run whest package --estimator estimator.py -o submission.tar.gz
```

This produces `submission.tar.gz` containing your `estimator.py`, the resolved `whestbench` version, and any imports your estimator needs (auto-detected).

## 📤 Submit to AIcrowd

Ship it straight from the CLI — no manual portal upload needed.

First, log in once with your AIcrowd API key (grab it from your
[AIcrowd profile](https://www.aicrowd.com/participants/me/customize)):

```bash
uv run whest login
```

Then submit. `whest submit` packages `estimator.py` and uploads it to the
challenge in one step (you can also submit a prebuilt tarball):

```bash
# package + submit in one go
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

- `estimator.py` — verbatim copy of yours
- `requirements.txt` — frozen from your `uv.lock`
- `metadata.json` — whestbench version, package timestamp

## After submission

What happens once `whest submit` (or a portal upload) accepts your
`submission.tar.gz`:

1. **AIcrowd unpacks the artifact** into a clean grader container that
   pre-installs the runner’s `whestbench` release plus the contents of
   your `requirements.txt`.
2. **The grader runs `whest run --runner docker`** against a held-out
   MLP suite (same `width`, `depth`, `flop_budget` as the public
   defaults; same `n_mlps` order of magnitude). No network, no GPU,
   no access to the local filesystem outside `SetupContext.scratch_dir`.
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
| `submission.tar.gz` size | Typically 2–10 KB without external deps; up to ~few MB with bundled wheels | If much larger, audit `requirements.txt` |
| Grader runtime | A few minutes for the default suite | Slower than that suggests `residual_wall_time_s` issues — see [score-report-fields.md](../reference/score-report-fields.md) |

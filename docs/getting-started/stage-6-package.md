# Stage 6: Package Your Submission

You've climbed the ladder. Now ship it.

## Run it

```bash
whest package --estimator estimator.py -o submission.tar.gz
```

This produces `submission.tar.gz` containing your `estimator.py`, the pinned `whestbench` version, and any imports your estimator needs (auto-detected). Upload that file to the AIcrowd submission portal.

## What's in the artifact

- `estimator.py` — verbatim copy of yours
- `requirements.txt` — frozen from your `uv.lock`
- `metadata.json` — whestbench version, package timestamp

## After submission

The AIcrowd evaluator re-runs `whest run --runner docker` on the same MLP suite. Your local Stage 4 score should match the leaderboard score within noise.

# TODOS — whest-starterkit

## v1.1
- [ ] `docs/CONTRIBUTING.md`: "How to add a new model class" walkthrough
      (covers surface contract, parity test pattern, RELEASING.md interaction).
      Why: design spec §10F — adding a new model is a 7-edit coordinated dance
      across 2 repos; needs to be a documented procedure.

- [ ] `tests/test_participant_flow.py`: end-to-end ladder test.
      (fresh tmpdir → uv sync → python estimator.py → whest validate → whest run).
      Why: README snapshot only covers fenced blocks; doesn't catch missed-step
      regressions across stages.

## Whestbench dependencies (not our work)
- Stage 5 (Docker runner) is "coming soon" until whestbench ships `--runner docker`.
  No fixed target. Update `docs/getting-started/stage-5-run-docker.md` placeholder
  when it lands.
- Bump-cron currently SHA-pins `whestbench` because no releases have been cut yet.
  Once whestbench cuts its first tag (e.g. `v0.1.0`), `.github/workflows/bump-whestbench.yml`
  will start preferring tags automatically (the `gh api .../releases/latest` call
  already handles both modes).

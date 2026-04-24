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

- [ ] **Asciinema demo cast** (Task 22 deferred from initial release).
      Record a 5-minute walkthrough on a real terminal:
      `make demo-cast` requires a TTY for proper styling and prompts.
      Then upload via `asciinema upload assets/demo.cast`, capture the asciinema.org
      ID, and replace the placeholder note at the top of `README.md` with:
      `[![asciicast](https://asciinema.org/a/<ID>.svg)](https://asciinema.org/a/<ID>)`.
      Optionally render to SVG locally with `agg` (`cargo install --git
      https://github.com/asciinema/agg`) and commit `assets/demo.svg` instead of
      relying on asciinema.org hosting.
      Why: design spec §6 (Delight #2) — visible "first 5 minutes" recording is
      the single highest-leverage README element for evaluating fit at a glance.

## Whestbench dependencies (not our work)
- Stage 5 (Docker runner) is "coming soon" until whestbench ships `--runner docker`.
  No fixed target. Update `docs/getting-started/stage-5-run-docker.md` placeholder
  when it lands.
- Bump-cron currently SHA-pins `whestbench` because no releases have been cut yet.
  Once whestbench cuts its first tag (e.g. `v0.1.0`), `.github/workflows/bump-whestbench.yml`
  will start preferring tags automatically (the `gh api .../releases/latest` call
  already handles both modes).

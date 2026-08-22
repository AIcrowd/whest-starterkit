# Releasing whest-starterkit

> [← Documentation](README.md)

Single source of truth for release operations. Keep this file current when the process changes.

## Cadence

The kit is not versioned, tagged, or published to PyPI; `main` is the live spec of the
current round. A "release" is a merge to `main` plus a dated entry in `CHANGELOG.md`.

## Routine release

1. **Verify CI passes.** All Stage 1-5 smokes pass before release.
2. **Refresh and review dependencies.** Run `uv lock` after dependency updates and inspect release notes when either `whestbench` or `flopscope` has semantically significant changes.
3. **Run `uv run python estimator.py` locally** to sanity-check the default template output after a dependency refresh. The header must read `MLP: width=1024 depth=16`.
4. **Banner sweep:** if the previous dated CHANGELOG entry added a "removed example" banner to README, delete it now (retire-after-1-release rule).
5. **Banner add:** if this release removes or renames any example or doc, add a banner block at the top of README:

   ```markdown
   > ⚠️ `examples/02_mean_propagation.py` was renamed to `examples/02_propagation_v2.py` on 2026-08-21. Update bookmarks.
   ```

   This banner stays for one release; the *next* release deletes it.
6. **Date the CHANGELOG.** Rename `## Unreleased` to today's date, and open a fresh empty
   `## Unreleased` above it. Anything that moved a rule, a parameter, a FLOP figure or the
   estimator contract belongs in that entry.
7. **Merge.**

## Dependency-pin bump (whestbench / flopscope)

1. whestbench publishes a release that names a flopscope minor version. Read its `pyproject.toml`.
2. Edit this repo's `pyproject.toml` so the whestbench floor equals whestbench's own version
   and the `flopscope` specifier string is **character-identical** to whestbench's.
3. `uv lock && uv sync --group dev`
4. Verify from whestbench's side:
   `uv run --with toml python <whestbench>/scripts/bump_starterkit_pin.py --check-deps`
5. Re-run the drift gates: `uv run pytest tests/ -q`
6. Re-measure the documented cost figures: `uv run python examples/02_mean_propagation.py`
   and `uv run python examples/03_covariance_propagation.py`.
7. Record the move in `CHANGELOG.md` under **Unreleased → Feat → deps**, naming what the
   evaluator pins at that moment (kit ahead of / level with / behind the grader).
8. If the round parameters changed, re-record `assets/demo.cast` (`make demo-cast-headless`).

## Surface-contract changes

If this PR changes any of:

- `BaseEstimator.predict` signature
- `MLP` dataclass field names
- `local_engine.{build_mlp, monte_carlo_layer_means, compare_against_monte_carlo}` signatures

→ It breaks working participant code. There is no version number to bump, so the announcement
*is* the release: file it under **BREAKING CHANGE** in the dated `CHANGELOG.md` entry, and
coordinate any new contract through [docs/reference/estimator-contract.md](reference/estimator-contract.md) until a separate contributing guide lands.

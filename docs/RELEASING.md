# Releasing whest-starterkit

Single source of truth for the bump dance. Stay current.

## Cadence

- **Patch** (`v1.0.0` → `v1.0.1`): doc fixes, typo fixes, example improvements.
- **Minor** (`v1.0.0` → `v1.1.0`): when whestbench bumps to a new minor.
- **Major** (`v1.0.0` → `v2.0.0`): when participant-visible workflow changes.

## Routine release (responding to a `bump-whestbench.yml` PR)

1. **Verify CI is green.** All Stage 1-3 smokes pass on the bump PR.
2. **Scan whestbench changelog.** If the upstream release notes mention any of {`BaseEstimator.predict` signature, `MLP` constructor, runner CLI flags}, this is a **major** bump and you need extra review.
3. **Run `python estimator.py` locally** against the bumped pin.
4. **Banner sweep:** if the *previous* release added a "removed example" banner to README, delete it now (retire-after-1-release rule).
5. **Banner add:** if this release removes or renames any example or doc, add a banner block at the top of README:

   ```markdown
   > ⚠️ `examples/02_mean_propagation.py` was renamed to `examples/02_propagation_v2.py` in vX.Y. Update bookmarks.
   ```

   This banner stays for ONE release, then is deleted by the *next* release.
6. **Merge.**
7. **Tag and push:**

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
8. **Cut a GitHub Release** with the changelog entry.

## Surface-contract changes (major bump)

If this PR changes any of:

- `BaseEstimator.predict` signature
- `MLP` dataclass field names
- `local_engine.{build_mlp, monte_carlo_layer_means, compare_against_monte_carlo}` signatures

→ This is a **major bump**. Update `docs/CONTRIBUTING.md` (when written — see TODOS.md v1.1) with any new contract.

## Cron failures

The `bump-whestbench.yml` workflow runs weekly. Status visible in the README badge.

- **Red badge after one firing:** transient (network, GitHub API). Wait for next firing.
- **Red badge after two firings:** investigate. Common causes: GitHub auth expired, whestbench renamed a tag, `uv lock` fails on a breaking-change pin.

We do NOT auto-create issues on cron failure. The badge is the signal.

## v1 launch sequence (one-time)

See spec §9.3 in `whestbench-public/.aicrowd/superpowers/specs/2026-04-24-whest-starterkit-design.md`.

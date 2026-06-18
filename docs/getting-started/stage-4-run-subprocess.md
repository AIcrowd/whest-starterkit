# Stage 4: Subprocess Runner

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · [2](stage-2-validate.md) · [3](stage-3-run-local.md) · **4** · [5](stage-5-package.md)

Stage 3 runs in your interpreter. Stage 4 spawns each estimator call in a fresh subprocess — the same isolation the grader uses. Catches:

- Shared global state between calls
- Stale RNG seeds from previous calls
- Memory leaks
- Imports that fail in a clean process

## 🚀 Run it

```bash
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess
```

Same score format as Stage 3. If your score drops noticeably, you've found a bug masked by in-process state.

## ✅ Expected outcome

Your Stage 4 `adjusted_final_layer_score` should match Stage 3 **exactly** — the
Mini split fixes the MLPs and bakes the ground truth at N=1e9, so there is no
Monte-Carlo noise between the two runs. If Stage 4 differs, you've found a bug
masked by in-process state.

If Stage 4 is **worse** than Stage 3, the most likely culprits are:
1. **Module-level mutable state** — `setup()` populated a global that
   persists between MLPs in-process but resets in subprocess workers.
2. **Caches keyed on object identity** — `id()` collisions in-process
   accidentally hit cached results; subprocess invalidates that.
3. **RNG seeded once at import time** — survives between in-process
   calls; subprocess re-seeds on every call.

Move state into the `Estimator` instance (or stash it on the
`SetupContext.scratch_dir`) and re-run.

## ✅ When you're ready

Move on to [Stage 5: Package your submission](stage-5-package.md).

# Stage 4: Subprocess runner

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · [2](stage-2-validate.md) · [3](stage-3-run-local.md) · **4** · [5](stage-5-package.md)

Stage 3 runs inside your `whest` process. Stage 4 hands your estimator to a separate
worker process over a pipe, the same transport the grader uses. **One worker serves the
whole suite**: `setup()` runs once, the module is imported once, and module-level state
carries across MLPs exactly as it does in Stage 3. What Stage 4 adds is a process
boundary, not a per-MLP state reset. It catches:

- Imports that only worked because your dev shell had something on `sys.path`
- Anything that writes to stdout — stdout *is* the IPC pipe; a stray `print()` in
  `predict()` desyncs the protocol and the run hangs until you kill it
- Values that do not survive the JSON round trip (predictions are serialized and re-read
  as float32)
- **Runaway memory, on Linux** — the worker sets an 8 GB `RLIMIT_AS` there, which is what
  the grader enforces. On macOS the call is refused
  (`[worker] could not setrlimit RLIMIT_AS=8192MB: current limit exceeds maximum limit`,
  which stays hidden unless the worker also dies) and the cap is not applied, so a Mac
  Stage 4 pass is **not** proof you fit in 8 GB.

It does **not** reset globals or RNG between MLPs. To reset them, clear per-MLP state at
the top of `predict()`, or loop `whest run --n-mlps 1 --seed k`.

(Stage 3 prints the notice `memory_limit_mb=8192 is advisory in --runner local` on every
run, which states the same fact from the other side: the 8 GB solution-process limit is
recorded but not enforced in-process. It is not a warning about your estimator.)

## 🚀 Run it

```bash
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner subprocess
```

Same score format as Stage 3.

## ✅ Expected outcome

Your Stage 4 `adjusted_final_layer_score` should match Stage 3 **exactly**. The
Mini split fixes the MLPs and bakes the ground truth at N=1e9, so there is no
Monte-Carlo noise between the two runs.

If Stage 4 is **worse** than Stage 3, the causes that can differ between the
two runners are:

1. **A `print()` or other stdout write** inside `setup()`/`predict()` — stdout is the
   worker's IPC channel. This does not merely fail: the run stalls, because the host is
   waiting for a JSON frame it will never parse. Kill it and remove the print (use
   `sys.stderr` or `logging`).
2. **An import that only resolves in your shell** — the worker starts fresh, so a helper
   module you never packaged raises `ImportError` here and not in Stage 3.
3. **Memory** — on Linux the worker caps itself at 8 GB via `RLIMIT_AS`, so a Stage 3 run
   that used 12 GB is killed there. whestbench replaces the dead worker (re-running
   your `setup()`) and continues, so you see a partial regression rather than a total
   failure. On macOS the cap is silently not applied (see above).
4. **Something that does not survive the pipe** — predictions cross as JSON and are re-read
   with `fnp.asarray(..., dtype=fnp.float32)`; return a `flopscope.numpy.ndarray` of shape
   `(depth, width)` and no other type.

State carried between MLPs is *not* on that list: one worker serves the suite, so a global
that `setup()` populated persists across MLPs under both runners. Reset per-MLP state at the
top of `predict()` rather than expecting the runner to do it. If you want a scratch path,
guard it: `whest run` leaves `ctx.scratch_dir` as `None` locally, so write
`base = ctx.scratch_dir or tempfile.mkdtemp()`. `ctx.submission_dir` *is* set locally (it
resolves to the directory holding your estimator file) and it contains your shipped
files.

## ✅ When you're ready

Move on to [Stage 5: package your submission](stage-5-package.md).

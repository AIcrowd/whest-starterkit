# Stage 2: Validate the Contract

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · **2** · [3](stage-3-run-local.md) · [4](stage-4-run-subprocess.md) · [5](stage-5-package.md)

Stage 1 confirms your estimator runs and converges. Stage 2 confirms it satisfies the harness contract: right shapes, right types, finite values, optional lifecycle hooks behave.

## 🚀 Run it

```bash
uv run whest validate --estimator estimator.py
```

A passing run renders a Rich panel with four green `OK` rows:

```
Validating estimator estimator.py
╭───────────────────────────────── Validation ─────────────────────────────────╮
│ Command: validate                                                            │
│ Status: success                                                              │
│ ╭───────────────────────────────── Checks ─────────────────────────────────╮ │
│ │ ┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓                   │ │
│ │ ┃ Status ┃ Check                    ┃ Detail         ┃                   │ │
│ │ ┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩                   │ │
│ │ │ OK     │ class resolved           │ Estimator      │                   │ │
│ │ │ OK     │ setup(context) completed │ 0.00s (cap 5s) │                   │ │
│ │ │ OK     │ predict() returned shape │ (2, 4)         │                   │ │
│ │ │ OK     │ values finite            │ all finite     │                   │ │
│ │ └────────┴──────────────────────────┴────────────────┘                   │ │
│ ╰──────────────────────────────────────────────────────────────────────────╯ │
╰──────────────────────────────────────────────────────────────────────────────╯
✓ Validation passed in 21ms
Next: `whest run --estimator estimator.py --split mini`
```

(The panel is drawn to your terminal width, and the trailing timing figure moves
run to run: ~20-30 ms here.)

## What each check does

- **`class resolved`** — the loader found `class Estimator(BaseEstimator)` in your file (override with `--class CustomName`).
- **`setup(context: SetupContext)`** — if you defined `setup()` (it's optional), the validator calls it with a probe `SetupContext` and confirms it returns without raising. See [SetupContext fields](../reference/estimator-contract.md#setupcontext-fields). The `0.00s (cap 5s)` detail is your `setup()` budget: it is the only limit whose breach fails the **whole** submission rather than one MLP, so rehearse a tighter one with `whest run --setup-timeout SECONDS`.
- **`predict() returned shape`** — invoked on a probe MLP (width=4, depth=2 by default) and asserts the returned array has shape `(mlp.depth, mlp.width)`.
- **`values finite`** — no NaN or Inf in the returned array.

A failing run halts at the first failed check and prints a structured error pointing at the contract clause that broke. See [estimator-contract.md](../reference/estimator-contract.md) for the full contract.

## ✅ Expected outcome

Validate is a sub-second sanity check against a probe MLP. It doesn't
score your estimator; it only checks that it returns the right *shape* of
the right *type* with finite values. Most participants pass on the first try.

If a check fails, the panel prints the broken clause inline. For shape
mismatches, the error includes `expected_shape` and `got_shape` so you
can fix the bug without reading source.

## ✅ When you're ready

Move on to [Stage 3: run the harness locally](stage-3-run-local.md).

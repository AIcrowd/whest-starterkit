# Stage 3: Run Locally (In-Process Harness)

Stage 2 confirms the contract. Stage 3 actually scores you against the same MLPs the grader uses — but in-process, so you can drop `import pdb; pdb.set_trace()` anywhere in `predict()` and step through it.

## Run it

```bash
whest run --estimator estimator.py --runner local
```

The default runner is `local` — you can omit `--runner local`. The output is a score report:

```
Score: 0.42
  - n_mlps:        20
  - mean MSE:      0.0014
  - flops used:    1.2e8 / 1.0e9 budget
```

## Why a different MSE than Stage 1?

Stage 1 uses one fixed MLP (`build_mlp(width=32, depth=6, seed=0)`). Stage 3 uses a **suite** of MLPs from a held-out dataset. Lower variance per MLP, but the average is what counts.

## Debugging

Because `--runner local` runs in-process, `pdb` works:

```python
def predict(self, mlp: MLP, budget: int) -> we.ndarray:
    import pdb; pdb.set_trace()
    ...
```

## When you're ready

Move on to [Stage 4: subprocess runner](stage-4-run-subprocess.md) for grader parity.

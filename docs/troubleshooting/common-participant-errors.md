# Common Participant Errors

Use this page when `validate` or `run` fails.

## 🛠 Estimator emitted wrong shape

Symptom: error mentions expected shape `(width,)`.

Why it happens: emitted `(1, width)` or `(width, 1)` instead of a 1D vector.

Fix now: ensure each yielded row is `np.ndarray` with shape `(circuit.n,)`.

Verify:

```bash
cestim validate --estimator ./my-estimator/estimator.py
```

## 🛠 Too few or too many rows

Symptom: error mentions row count or `max_depth` mismatch.

Why it happens: stopping early or yielding extra rows.

Fix now: yield exactly one row per layer, no more and no less.

Verify:

```bash
cestim validate --estimator ./my-estimator/estimator.py
```

## 🛠 Non-finite values (`nan` or `inf`)

Symptom: error mentions finite values.

Why it happens: unstable numeric operations.

Fix now: add guards/clipping/checks before yielding rows.

Verify:

```bash
cestim validate --estimator ./my-estimator/estimator.py
```

## 🛠 Non-iterable `predict` output

Symptom: error indicates estimator output is not an iterator.

Why it happens: returning a scalar/tensor instead of yielding rows.

Fix now: implement `predict` as a generator and `yield` each depth row.

Verify:

```bash
cestim validate --estimator ./my-estimator/estimator.py
```

## 🛠 Runtime envelope penalties

Symptom: unexpectedly poor `adjusted_mse` despite reasonable `mse_mean`.

Why it happens: depth rows timing outside tolerance bounds.

Fix now:

- simplify compute for smaller budgets,
- keep per-depth runtime stable,
- compare `mse_mean` vs `adjusted_mse` to diagnose timing penalties.

Verify:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner subprocess --json
```

## ➡️ Next step

- [Estimator Contract](../reference/estimator-contract.md)
- [Scoring Model](../concepts/scoring-model.md)

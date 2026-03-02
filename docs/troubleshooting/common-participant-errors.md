# Common Participant Errors

## Estimator emitted wrong shape

Symptom:

- error mentions expected shape `(width,)`.

Likely cause:

- emitted `(1, width)` or `(width, 1)` instead of a 1D vector.

Fix:

- ensure each yielded row is exactly `np.ndarray` with shape `(circuit.n,)`.

## Too few or too many rows

Symptom:

- error mentions row count or `max_depth` mismatch.

Likely cause:

- stopping early or yielding extra rows.

Fix:

- yield exactly one row per layer, no more and no less.

## Non-finite values (`nan` or `inf`)

Symptom:

- error mentions finite values.

Likely cause:

- unstable numeric operations.

Fix:

- add guards/clipping/checks before yielding rows.

## Non-iterable `predict` output

Symptom:

- error indicates estimator output is not an iterator.

Likely cause:

- returning a scalar/tensor instead of yielding rows.

Fix:

- implement `predict` as a generator and `yield` each depth row.

## Runtime envelope penalties

Symptom:

- unexpectedly poor `adjusted_mse` despite reasonable `mse_mean`.

Likely cause:

- depth rows timing outside tolerance bounds.

Fix:

- simplify compute for smaller budgets,
- keep per-depth runtime stable,
- compare `mse_mean` vs `adjusted_mse` to diagnose timing penalties.

## Next

- [Estimator Contract](../reference/estimator-contract.md)
- [Scoring Model](../concepts/scoring-model.md)

# Code patterns

> [← Documentation](../README.md)

Quick reference for flopscope operations. All examples assume `import flopscope as flops
import flopscope.numpy as fnp`.

## Operators are tracked

Python arithmetic operators (`+`, `-`, `*`, `/`, `@`) on `fnp.ndarray` values are
FLOP-tracked. You do not need the verbose `fnp.add` and `fnp.multiply` forms.

```python
import flopscope as flops
import flopscope.numpy as fnp

a = fnp.ones(4)
b = fnp.ones(4)

# These are all equivalent and all tracked:
c = a + b           # tracked: same as fnp.add(a, b)
d = a * b           # tracked: same as fnp.multiply(a, b)
e = a / b           # tracked: same as fnp.divide(a, b)

W = fnp.eye(4)
v = fnp.ones(4)
f = W @ v           # tracked: same as fnp.matmul(W, v)
g = W.T @ v         # tracked: transpose is free, matmul is tracked
```

Use operators whenever they improve readability. The verbose `fnp.*` forms are still
available but are no longer required for tracking purposes.

### Avoid chained matmuls — they drop symmetry information

flopscope tracks symmetry annotations on tensors. Operations that produce a
mathematically-symmetric result will tag the output as symmetric **only if
flopscope can prove it from the operands and the operation**. Chained
matmuls (`A @ B @ C`) defeat this proof because each `matmul` runs in
isolation: the intermediate `(A @ B)` is generally not symmetric, so the
final `@ C` can't recover symmetry even when the full triple product
mathematically is.

The canonical example is the covariance update inside a linear layer:

```python
# Anti-pattern — flopscope cannot prove cov_pre is symmetric,
# downstream multiplies emit SymmetryLossWarning:
cov_pre = w.T @ cov @ w

# Use einsum (not the chained `w.T @ cov @ w`) so the whole sandwich is one
# contraction over a symmetric-tagged `cov`. Since flopscope 0.10.0 you must
# also re-tag after any write into cov — a write voids the tag rather than
# letting a stale claim keep its discount.
cov = fnp.eye(width, dtype=fnp.float32)   # already tagged — fnp.zeros/ones/eye on
                                          # a square shape return a SymmetricTensor
for w in mlp.weights:
    cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)   # symmetric rate
    ...
    cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)  # <- voids the tag
    fnp.fill_diagonal(cov, var_post)                    # <- and so does this
    cov = flops.as_symmetric(cov, symmetry=(0, 1))      # re-validate, ~7n FLOPs
```

You will **not** get a warning here. `fill_diagonal` voids the tag silently, and
the `multiply` above it keeps the tag because `outer(gain, gain)` is itself
symmetry-inferred, so nothing tells you the discount has gone. Re-tag
unconditionally after any write into `cov`. To confirm it worked, call
`ctx.summary()`: the einsum should read 3,220,700,672 per layer, not
4,292,870,144. (`SymmetryLossWarning` does exist: it fires when a tagged
operand meets one that shares no symmetry group, as in the anti-pattern above,
or on a reshape across the symmetric block.)

The re-tag is cheap next to the contraction it keeps on the symmetric rate: at
the Phase 2 shape (`width=1024, depth=16`), the 16 per-layer `as_symmetric`
calls in `examples/03_covariance_propagation.py` cost 117,440,496 FLOPs between
them (0.23% of that estimator's bill), against 51,531,210,752 spent on the
einsums themselves. (`as_symmetric` bills `7n-1` per element-pass with
`n = width²`, so 7,340,031 FLOPs a call at the float32 rate the example seeds
its covariance in; float64 would bill 2×. The example also wraps its seed identity
in `as_symmetric`, which is redundant for the reason above; that is why its own
summary reads 17 calls / 124,780,527.)

See `examples/03_covariance_propagation.py` for the full pattern in
context, and [whestbench#27](https://github.com/AIcrowd/whestbench/issues/27)
for the rationale.

## Operation costs

> **flopscope 0.12 billing model** (the version this kit pins). Costs are
> `flop_cost × weight × dtype_rate` (× a complex factor for complex dtypes).
> Practical rules: stay in float32 (float64 bills 2×); write `x * x` not
> `x ** 2` (power is 16-tier); `zeros` and views are free but
> `ones`/`eye`/`stack`/`concatenate`/copies now bill 1×/element; gathers and
> 3-arg `where` bill 4×/element; sorts bill ≈4·N·⌈log₂N⌉ (per comparison).
> Since 0.11.0 a non-numeric dtype (`object`, `str_`, `bytes_`, `datetime64`,
> structured) raises `UnsupportedDtypeError` at any metered op, free ones
> included, so `fnp.array([1.0, None])` and `fnp.zeros(3, dtype=object)` now
> fail rather than bill; convert with plain `numpy` first. Full audited tables:
> [flopscope cost model](https://aicrowd.github.io/flopscope/docs/understanding/flop-counting-model/).

| What you want | Code | FLOP cost | Notes |
|---|---|---|---|
| Create zeros | `fnp.zeros((n, n))` | 0 | Free |
| Create ones | `fnp.ones(n)` | 1 per element | Billed since 0.9 |
| Identity matrix | `fnp.eye(n)` | n (diagonal only) | Billed since 0.9 |
| Wrap existing data | `fnp.asarray(data)` | 0 | Free if no copy needed; `fnp.array()` always copies |
| Matrix multiply | `fnp.matmul(A, B)`, `A @ B` | `M·N·(2K−1)` for `(M,K) @ (K,N)` | ~2·M·N·K — K multiplies plus K−1 adds per output element. At `(1024,1024) @ (1024,1024)` float32 that is **2,146,435,072**, so `2**41` buys 1,024 of them, not 2,048. Dominates budgets. |
| Element-wise add | `fnp.add(a, b)` | 1 per element | |
| Element-wise multiply | `fnp.multiply(a, b)` | 1 per element | |
| Element-wise divide | `fnp.divide(a, b)` | 1 per element | |
| ReLU | `fnp.maximum(x, 0.0)` | 1 per element | |
| Square root | `fnp.sqrt(x)` | 1 per element | |
| Exponential | `fnp.exp(x)` | 16 per element | Transcendental — 16x tier |
| Logarithm | `fnp.log(x)` | 16 per element | Transcendental — 16x tier |
| Transpose | `fnp.transpose(W)` | 0 | Free |
| Reshape | `fnp.reshape(x, shape)` | 1 per element | Billed since 0.9 (materializes) |
| Extract diagonal | `fnp.diag(M)` | 0 | Free |
| Set diagonal | `fnp.fill_diagonal(M, v)` | n (diagonal only) | Billed since 0.9, in-place |
| Outer product | `fnp.outer(a, b)` | n x m | |
| Sum / Max / Min | `fnp.sum(x, axis=0)`, `fnp.max(x)` | `numel − n_outputs` | One combine per step; ≈ input size |
| Mean | `fnp.mean(x, axis=0)` | `numel` | The sum (`numel − n_outputs`), plus one divide per output |
| Stack arrays | `fnp.stack(rows, axis=0)` | 1 per element | Billed since 0.9 |
| Concatenate | `fnp.concatenate([a, b])` | 1 per element | Billed since 0.9 |
| Index/slice | `x[0]`, `x[:, 3]` | 0 | Free |

## Common patterns

### Seed randomness from `mlp.seed` and `ctx.seed`

The grader supplies two independent seeds: `mlp.seed` for per-MLP randomness inside `predict()`, and `ctx.seed` for run-level randomness inside `setup()` (the same value on every `setup()` call in the run). Use them for any RNG inside your estimator.

**Predict-time** (per-MLP randomness):

```python
import flopscope.numpy as fnp

def predict(self, mlp, budget):
    rng = fnp.random.default_rng(mlp.seed)
    samples = rng.standard_normal((n_samples, mlp.width))
    ...
```

For multiple independent RNG streams within one `predict()` call, spawn sub-generators from the per-MLP root rather than choosing your own seeds:

```python
master = fnp.random.default_rng(mlp.seed)
sub_a, sub_b, sub_c = (
    fnp.random.default_rng(s)
    for s in master.bit_generator.spawn(3)
)
```

**Setup-time** (run-level randomness, such as fixed random projections):

```python
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext

class Estimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        self.setup_rng = fnp.random.default_rng(ctx.seed)
        # one-time precompute, e.g. a (width, k) random projection basis
        self.projection = self.setup_rng.standard_normal((ctx.width, 64))
```

`setup()` runs once per worker process (several times per submission) under a
hard 5 s ceiling each, so keep it to work of that size and make it idempotent.
A projection basis fits comfortably; anything heavier belongs in a shipped data
file ([Ship Weights](../how-to/ship-weights.md)). See
[Estimator Contract: Lifecycle](./estimator-contract.md#lifecycle).

Do **not** call `fnp.random.seed(ctx.seed)`: it mutates the process-global RNG. Always use `fnp.random.default_rng(...)` for an isolated `Generator`.

Participant-chosen seeds (for example, `fnp.random.default_rng(42)` inside `predict()` or `setup()`) may be disqualified for prize eligibility; see [Estimator Contract: Reproducibility under the grader seed](./estimator-contract.md#reproducibility-under-the-grader-seed).

### Standard normal PDF and CDF (built-in)

flopscope provides built-in PDF and CDF functions that are FLOP-tracked:

```python
import flopscope as flops
import flopscope.numpy as fnp

phi = flops.stats.norm.pdf(x)   # standard normal PDF
Phi = flops.stats.norm.cdf(x)   # standard normal CDF
```

These are the built-ins, and the two propagation examples
([`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py) and
[`examples/03_covariance_propagation.py`](../../examples/03_covariance_propagation.py))
both use them. They cost more than the hand-rolled versions below
(`flops.stats.norm.pdf` ≈ 54/element against ≈ 20, `.cdf` ≈ 96/element against
≈ 48) and promote float32 input to float64 to match scipy, so pick deliberately.

### Standard normal PDF (for ReLU expectation)

```python
import flopscope as flops
import flopscope.numpy as fnp

def norm_pdf(x):
    """phi(x) = exp(-x^2/2) / sqrt(2*pi)"""
    return fnp.exp(-0.5 * x * x) / fnp.sqrt(2.0 * fnp.pi)
```

### Standard normal CDF

Pure flopscope implementation using the Abramowitz & Stegun approximation (accurate to <7.5e-8):

```python
import flopscope as flops
import flopscope.numpy as fnp

_P = 0.2316419
_A1, _A2, _A3 = 0.319381530, -0.356563782, 1.781477937
_A4, _A5 = -1.821255978, 1.330274429

def norm_cdf(x):
    t = 1.0 / (1.0 + _P * fnp.abs(x))
    poly = ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    pdf = fnp.exp(-0.5 * x * x) / fnp.sqrt(2.0 * fnp.pi)
    cdf = 1.0 - pdf * poly
    return fnp.where(x >= 0, cdf, 1.0 - cdf)
```

> Use the pure-flopscope version above. The grader sandbox provides only
> `flopscope`, the `whestbench` API, and the Python standard library. `scipy`
> and every other third-party PyPI package are absent, and only flopscope
> operations are FLOP-counted. Do not route around that by vendoring
> `numpy`/`scipy`/a BLAS into your submission, or by using compiled kernels or
> `ctypes`/`cffi`: those are prohibited and are grounds for disqualification.
> Shipping *data* files (weights, lookup tables, precomputed artifacts) remains
> permitted; see [Ship Weights](../how-to/ship-weights.md) and
> [Allowed Code](../concepts/allowed-code.md).

### ReLU expectation (E[max(0, z)] where z ~ N(mu, sigma^2))

```python
import flopscope as flops
import flopscope.numpy as fnp

alpha = mu_pre / sigma_pre
E_relu = mu_pre * norm_cdf(alpha) + sigma_pre * norm_pdf(alpha)
```

#### Why this works

`ReLU(z) = max(z, 0)` zeros out everything below 0 and keeps everything
above. If `z ~ N(µ, σ²)`, the expectation splits into the part above zero
and the part below (which contributes 0):

```
E[ReLU(z)] = ∫_0^∞ z · f(z) dz
           = µ · Φ(α) + σ · φ(α)        where α = µ / σ
```

Here `Φ` is the standard-normal CDF, `φ` is the standard-normal PDF, and
`α` measures how many standard deviations the mean sits above zero.
`µ · Φ(α)` is the contribution of the probability mass above zero, and
`σ · φ(α)` corrects for the mass that the rectification clips at zero. This is
the (rectified Gaussian) first moment; see Frey & Hinton (1999) and Williams
(1998) for derivations.

#### Where the assumption breaks

The pre-activation `z` is exactly Gaussian only at layer 0. After that,
every layer is `W·ReLU(prev)`, and the resulting distribution is Gaussian
only by approximation (Central Limit Theorem on the matmul gives a good
fit for moderate widths). The approximation degrades when:

- **Widths are small.** CLT averaging is weak below ~32 neurons per layer.
- **Networks are very deep.** Errors compound layer-by-layer; by depth
  ~32 you may want higher moments (skewness) or per-layer recalibration.
- **Activations cluster near zero.** When `α ≈ 0`, the rectified-Gaussian
  approximation is accurate, but `µ` is small and relative errors spike.

The Phase 2 shape (`width=1024, depth=16`) avoids the first two: 1024 neurons
per layer is wide enough for the CLT argument to hold, and 16 layers is few
enough to limit compounding.

If your `final_layer_mse` is low but `all_layers_mse` is high, this assumption
is the usual cause. See [algorithm-ideas.md](../how-to/algorithm-ideas.md)
for advanced moment-matching strategies.

See [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py) for a complete working estimator using these patterns.

### Per-neuron variance propagation (diagonal)

```python
import flopscope as flops
import flopscope.numpy as fnp

# var_pre[i] = sum_j W[j,i]^2 * var[j]
var_pre = (w * w).T @ var
```

## ➡️ Next step

- [Manage Your FLOP Budget](../how-to/manage-flop-budget.md)
- [Algorithm Ideas](../how-to/algorithm-ideas.md)
- [Estimator Contract](./estimator-contract.md)

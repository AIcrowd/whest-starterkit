# Code Patterns

> [← Documentation](../README.md)

Quick reference for flopscope operations. All examples assume `import flopscope as flops
import flopscope.numpy as fnp`.

## Operators are tracked

Python arithmetic operators (`+`, `-`, `*`, `/`, `@`) on `fnp.ndarray` values are
FLOP-tracked — you do not need to use the verbose `fnp.add`, `fnp.multiply`, etc. forms.

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
isolation — the intermediate `(A @ B)` is generally not symmetric, so the
final `@ C` can't recover symmetry even when the full triple product
mathematically is.

The canonical example is the covariance update inside a linear layer:

```python
# Anti-pattern — flopscope cannot prove cov_pre is symmetric,
# downstream multiplies emit SymmetryLossWarning:
cov_pre = w.T @ cov @ w

# Use einsum so flopscope sees both `w` operands are the same tensor
# and tags cov_pre as symmetric. Symmetry then flows downstream:
cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
```

See `examples/03_covariance_propagation.py` for the full pattern in
context, and [whestbench#27](https://github.com/AIcrowd/whestbench/issues/27)
for the rationale.

## Operation costs

| What you want | Code | FLOP cost | Notes |
|---|---|---|---|
| Create zeros | `fnp.zeros((n, n))` | 0 | Free |
| Create ones | `fnp.ones(n)` | 0 | Free |
| Identity matrix | `fnp.eye(n)` | 0 | Free |
| Wrap existing data | `fnp.array(data)` | 0 | Free |
| Matrix multiply | `fnp.matmul(A, B)` | O(m x n x k) | Dominates budgets |
| Element-wise add | `fnp.add(a, b)` | 1 per element | |
| Element-wise multiply | `fnp.multiply(a, b)` | 1 per element | |
| Element-wise divide | `fnp.divide(a, b)` | 1 per element | |
| ReLU | `fnp.maximum(x, 0.0)` | 1 per element | |
| Square root | `fnp.sqrt(x)` | 1 per element | |
| Exponential | `fnp.exp(x)` | 1 per element | |
| Logarithm | `fnp.log(x)` | 1 per element | |
| Transpose | `fnp.transpose(W)` | 0 | Free |
| Reshape | `fnp.reshape(x, shape)` | 0 | Free |
| Extract diagonal | `fnp.diag(M)` | 0 | Free |
| Set diagonal | `fnp.fill_diagonal(M, v)` | 0 | Free, in-place |
| Outer product | `fnp.outer(a, b)` | n x m | |
| Sum | `fnp.sum(x, axis=0)` | input size | |
| Mean | `fnp.mean(x, axis=0)` | input size | |
| Max | `fnp.max(x)` | input size | |
| Stack arrays | `fnp.stack(rows, axis=0)` | 0 | Free |
| Concatenate | `fnp.concatenate([a, b])` | 0 | Free |
| Index/slice | `x[0]`, `x[:, 3]` | 0 | Free |

## Common patterns

### Seed randomness from `mlp.seed` and `ctx.seed`

The grader supplies two independent seeds: `mlp.seed` for per-MLP randomness inside `predict()`, and `ctx.seed` for one-time randomness inside `setup()`. Use them for any RNG inside your estimator.

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

**Setup-time** (run-level randomness, e.g. fixed random projections):

```python
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext

class Estimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        self.setup_rng = fnp.random.default_rng(ctx.seed)
        # one-time precompute, e.g. a (width, k) random projection basis
        self.projection = self.setup_rng.standard_normal((ctx.width, 64))
```

Do **not** call `fnp.random.seed(ctx.seed)` — that mutates the process-global RNG. Always use `fnp.random.default_rng(...)` for an isolated `Generator`.

Participant-chosen seeds (e.g. `fnp.random.default_rng(42)` inside `predict()` or `setup()`) may be disqualified for prize eligibility — see [Estimator Contract: Reproducibility](./estimator-contract.md#reproducibility-under-the-grader-seed).

### Standard normal PDF and CDF (built-in)

flopscope provides built-in PDF and CDF functions that are FLOP-tracked:

```python
import flopscope as flops
import flopscope.numpy as fnp

phi = flops.stats.norm.pdf(x)   # standard normal PDF
Phi = flops.stats.norm.cdf(x)   # standard normal CDF
```

These are the recommended approach — all example estimators use them. The manual implementations below are shown for reference.

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

> Use the pure-flopscope version above. The grader sandbox does **not** provide
> `scipy` (or any third-party PyPI package) — only `flopscope`, the `whestbench`
> API, and the Python standard library are importable — and only flopscope
> operations are FLOP-counted.

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
Intuitively: `µ · Φ(α)` is "what survives if the distribution is mostly
positive"; `σ · φ(α)` is the "edge correction" for the part of the bell
that's clipped at zero. This is the (rectified Gaussian) first moment;
see e.g. Frey & Hinton (1999), Williams (1998) for derivations.

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

If your `final_layer_mse` is fine but `all_layers_mse` blows up, this assumption
is usually the culprit. See [algorithm-ideas.md](../how-to/algorithm-ideas.md)
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

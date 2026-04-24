# Code Patterns

Quick reference for whest operations. All examples assume `import whest as we`.

## Operators are tracked

Python arithmetic operators (`+`, `-`, `*`, `/`, `@`) on `we.ndarray` values are
FLOP-tracked — you do not need to use the verbose `we.add`, `we.multiply`, etc. forms.

```python
import whest as we

a = we.ones(4)
b = we.ones(4)

# These are all equivalent and all tracked:
c = a + b           # tracked: same as we.add(a, b)
d = a * b           # tracked: same as we.multiply(a, b)
e = a / b           # tracked: same as we.divide(a, b)

W = we.eye(4)
v = we.ones(4)
f = W @ v           # tracked: same as we.matmul(W, v)
g = W.T @ v         # tracked: transpose is free, matmul is tracked
h = W.T @ W @ v     # tracked: two matmuls, chained with @
```

Use operators whenever they improve readability. The verbose `we.*` forms are still
available but are no longer required for tracking purposes.

## Operation costs

| What you want | Code | FLOP cost | Notes |
|---|---|---|---|
| Create zeros | `we.zeros((n, n))` | 0 | Free |
| Create ones | `we.ones(n)` | 0 | Free |
| Identity matrix | `we.eye(n)` | 0 | Free |
| Wrap existing data | `we.array(data)` | 0 | Free |
| Matrix multiply | `we.matmul(A, B)` | O(m x n x k) | Dominates budgets |
| Element-wise add | `we.add(a, b)` | 1 per element | |
| Element-wise multiply | `we.multiply(a, b)` | 1 per element | |
| Element-wise divide | `we.divide(a, b)` | 1 per element | |
| ReLU | `we.maximum(x, 0.0)` | 1 per element | |
| Square root | `we.sqrt(x)` | 1 per element | |
| Exponential | `we.exp(x)` | 1 per element | |
| Logarithm | `we.log(x)` | 1 per element | |
| Transpose | `we.transpose(W)` | 0 | Free |
| Reshape | `we.reshape(x, shape)` | 0 | Free |
| Extract diagonal | `we.diag(M)` | 0 | Free |
| Set diagonal | `we.fill_diagonal(M, v)` | 0 | Free, in-place |
| Outer product | `we.outer(a, b)` | n x m | |
| Sum | `we.sum(x, axis=0)` | input size | |
| Mean | `we.mean(x, axis=0)` | input size | |
| Max | `we.max(x)` | input size | |
| Stack arrays | `we.stack(rows, axis=0)` | 0 | Free |
| Concatenate | `we.concatenate([a, b])` | 0 | Free |
| Index/slice | `x[0]`, `x[:, 3]` | 0 | Free |

## Common patterns

### Standard normal PDF and CDF (built-in)

whest provides built-in PDF and CDF functions that are FLOP-tracked:

```python
import whest as we

phi = we.stats.norm.pdf(x)   # standard normal PDF
Phi = we.stats.norm.cdf(x)   # standard normal CDF
```

These are the recommended approach — all example estimators use them. The manual implementations below are shown for reference.

### Standard normal PDF (for ReLU expectation)

```python
import whest as we

def norm_pdf(x):
    """phi(x) = exp(-x^2/2) / sqrt(2*pi)"""
    return we.exp(-0.5 * x * x) / we.sqrt(2.0 * we.pi)
```

### Standard normal CDF

Pure whest implementation using the Abramowitz & Stegun approximation (accurate to <7.5e-8):

```python
import whest as we

_P = 0.2316419
_A1, _A2, _A3 = 0.319381530, -0.356563782, 1.781477937
_A4, _A5 = -1.821255978, 1.330274429

def norm_cdf(x):
    t = 1.0 / (1.0 + _P * we.abs(x))
    poly = ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    pdf = we.exp(-0.5 * x * x) / we.sqrt(2.0 * we.pi)
    cdf = 1.0 - pdf * poly
    return we.where(x >= 0, cdf, 1.0 - cdf)
```

Alternatively, if you add `scipy` to your `requirements.txt`:

```python
# Optional: requires scipy as a user-provided dependency
from scipy.special import ndtr

def norm_cdf(x):
    return we.array(ndtr(we.asarray(x, dtype=we.float64)).astype(we.float32))
```

### ReLU expectation (E[max(0, z)] where z ~ N(mu, sigma^2))

```python
import whest as we

alpha = mu_pre / sigma_pre
E_relu = mu_pre * norm_cdf(alpha) + sigma_pre * norm_pdf(alpha)
```

See [`examples/estimators/mean_propagation.py`](../../examples/estimators/mean_propagation.py) for a complete working estimator using these patterns.

### Per-neuron variance propagation (diagonal)

```python
import whest as we

# var_pre[i] = sum_j W[j,i]^2 * var[j]
var_pre = (w * w).T @ var
```

## Next step

- [Manage Your FLOP Budget](../how-to/manage-flop-budget.md)
- [Algorithm Ideas](../how-to/algorithm-ideas.md)
- [Estimator Contract](./estimator-contract.md)

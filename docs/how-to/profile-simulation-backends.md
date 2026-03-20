# Profile Simulation Backends

## When to use this page

The starter kit ships with multiple simulation backends — NumPy, PyTorch, Numba, SciPy, JAX, and Cython — each with different performance characteristics depending on your hardware. The `profile-simulation` command lets you benchmark them head-to-head so you can pick the fastest option for your machine.

Use this page when you want to:

- **Find the fastest backend for your hardware** — performance varies dramatically between Apple Silicon, x86 laptops, and cloud instances.
- **Verify a backend is installed and correct** — the profiler runs a pre-flight correctness check before timing.
- **Collect reproducible benchmark data** — JSON output includes hardware metadata and library versions.

## Do this now

### 1. Install backends

By default, only NumPy and SciPy are available. **You must install optional backends before the profiler can use them.** The fastest way is the `all-backends` dependency group:

```bash
uv sync --group all-backends
```

This installs PyTorch, Numba, JAX, and Cython. See [Installing optional backends](#installing-optional-backends) below for per-backend instructions and the extra build step required for Cython.

> **Why do I only see numpy and scipy?** You probably haven't run `uv sync --group all-backends` yet. The profiler can only benchmark backends that are installed in your environment.

### 2. Run a quick benchmark

```bash
nestim profile-simulation --preset quick
```

This finishes in seconds and gives you a first look at which backends are available, which pass correctness, and rough speedup numbers.

### 3. Run the standard benchmark

```bash
nestim profile-simulation
```

The default `standard` preset tests two widths (64, 256), five depths (4–128), and three sample counts (10k–1M). It takes a few minutes and gives a reliable picture of relative performance.

### 4. Save results for comparison

```bash
nestim profile-simulation --output results.json
```

The JSON file contains everything needed to compare runs across machines:

```json
{
  "hardware": {
    "platform": "Linux-5.15.0-aws-x86_64",
    "processor": "x86_64",
    "cpu_count": 4,
    "python_version": "3.11.7",
    "machine": "x86_64"
  },
  "backend_versions": {
    "numpy": "1.26.4",
    "pytorch": "2.2.1"
  },
  "skipped_backends": {
    "jax": "pip install 'jax[cpu]>=0.4'"
  },
  "correctness": [
    {"backend": "numpy", "passed": true, "error": ""},
    {"backend": "pytorch", "passed": true, "error": ""}
  ],
  "timing": [
    {
      "backend": "pytorch",
      "operation": "run_mlp",
      "width": 256,
      "depth": 32,
      "n_samples": 100000,
      "times": [0.1234, 0.1201, 0.1215],
      "median_time": 0.1215,
      "speedup_vs_numpy": 2.45
    }
  ]
}
```

## Choosing presets

| Preset | Widths | Depths | Sample counts | Typical time |
|--------|--------|--------|---------------|--------------|
| `super-quick` | 64 | 4 | 1k | Sub-second |
| `quick` | 256 | 4, 32 | 10k, 100k | Seconds |
| `standard` | 64, 256 | 4, 16, 32, 64, 128 | 10k, 100k, 1M | Minutes |
| `exhaustive` | 64, 128, 256 | 4, 16, 32, 64, 128 | 10k–16.7M | Long |

Use `quick` for a fast sanity check, `standard` for development decisions, and `exhaustive` when you need thorough data (e.g. before choosing a backend for production scoring).

## Filtering backends

Profile only the backends you care about:

```bash
# Only NumPy and PyTorch
nestim profile-simulation --backends numpy,pytorch

# Only Numba
nestim profile-simulation --backends numba
```

Valid backend names: `numpy`, `pytorch`, `numba`, `scipy`, `jax`, `cython`.

## Installing optional backends

Only NumPy and SciPy are installed by default. **The profiler will skip any backend that isn't installed** and show install hints in the output:

```text
Skipped backends:
  pytorch: not installed. Install: pip install torch>=2.0
  numba: not installed. Install: pip install numba>=0.58
  jax: not installed. Install: pip install 'jax[cpu]>=0.4'
  cython: not installed. Install: pip install cython>=3.0 && python setup_cython.py build_ext --inplace
```

### Install all backends at once (recommended)

```bash
uv sync --group all-backends
```

This installs PyTorch, Numba, JAX, and Cython via the `all-backends` dependency group defined in `pyproject.toml`. Note that **Cython also requires a build step** — see [Building the Cython backend](#building-the-cython-backend) below.

### Install individual backends

```bash
pip install torch>=2.0          # PyTorch
pip install numba>=0.58         # Numba
pip install 'jax[cpu]>=0.4'     # JAX
pip install cython>=3.0         # Cython (step 1 of 2 — also needs build step)
```

### Building the Cython backend

The Cython backend uses compiled C extensions with direct BLAS calls for maximum speed. Unlike other backends, installing the `cython` package alone is **not enough** — you must also compile the extension module.

**Step 1: Install Cython**

```bash
uv sync --group cython
# or: pip install cython>=3.0
```

**Step 2: Build the extension**

From the project root directory:

```bash
python setup_cython.py build_ext --inplace
```

This compiles `src/network_estimation/_cython_kernels.pyx` into a shared library that Python can import. The build requires:

- A C compiler (gcc, clang, or MSVC)
- NumPy headers (included automatically via the installed numpy package)
- SciPy (for BLAS function declarations used by the Cython code)

**Verify it works:**

```bash
nestim profile-simulation --preset super-quick --backends cython
```

If the build succeeded, you should see `cython` in the correctness check and timing results. If it shows as skipped, re-run the build step and check for compiler errors.

> **Troubleshooting Cython build failures:**
> - **"command 'gcc' not found"** — Install a C compiler. On macOS: `xcode-select --install`. On Ubuntu: `apt install build-essential`.
> - **"numpy/arrayobject.h: No such file"** — NumPy headers are missing. Re-install numpy: `pip install --force-reinstall numpy`.
> - **"cannot import name '_cython_kernels'"** — The `.so`/`.pyd` file wasn't placed correctly. Make sure you run the build command from the project root (where `setup_cython.py` lives).

## Selecting a backend for scoring

Once you've identified the fastest backend, set the `NESTIM_BACKEND` environment variable so that `nestim run` and `nestim create-dataset` use it:

```bash
export NESTIM_BACKEND=pytorch
nestim run --estimator ./my-estimator/estimator.py
```

Valid values match the backend names: `numpy`, `pytorch`, `numba`, `scipy`, `jax`, `cython`. The default is `numpy`.

## Understanding the output

The terminal table shows:

- **Skipped backends** — not installed, with install commands.
- **Pre-flight Correctness Check** — PASS or FAIL for each backend. Failed backends are excluded from timing.
- **Timing Results** — one row per (backend, operation, width, depth, n_samples) combination:
  - **Median Time (s)** — median wall-clock time across 3 iterations.
  - **Speedup vs NumPy** — how many times faster than NumPy. Green = faster, red = slower.
  - **Status** — OK if the backend passed correctness, FAIL otherwise.

## Common workflows

### Compare performance across machines

Run the same command on each machine and save JSON:

```bash
# On your laptop
nestim profile-simulation --preset standard --output laptop.json

# On your AWS instance
nestim profile-simulation --preset standard --output aws-c5-4xlarge.json
```

Then compare the JSON files to find the best backend for each environment.

### Debug a failing backend

If a backend shows FAIL in the correctness check:

```bash
# Run just that backend with debug output
nestim profile-simulation --backends pytorch --debug
```

The error message in the correctness output will indicate whether the issue is numerical (tolerance failure) or a missing dependency.

## ✅ Expected outcome

- The terminal displays a formatted table with correctness and timing results.
- Backends with missing dependencies are listed with install hints.
- If `--output` is provided, a JSON file is written with full hardware metadata.
- Speedup values > 1.0 indicate backends faster than NumPy.

## ➡️ Next step

- [CLI Reference: profile-simulation](../reference/cli-reference.md#nestim-profile-simulation) — full flag reference
- [Use Evaluation Datasets](./use-evaluation-datasets.md) — pre-create datasets for faster iteration
- [Validate, Run, and Package](./validate-run-package.md) — score your estimator

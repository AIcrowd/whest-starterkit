# Use WhestBench Explorer

![WhestBench Explorer -- a small network with 4 neurons and 5 layers, after running Ground Truth estimation](../../assets/whestbench-explorer-visualization.svg)

## When to use this page

Use this page when you want visual intuition about network behavior and estimator error patterns.

WhestBench Explorer is optional and is not the submission interface.

## Do this now

```bash
whest visualizer
```

This checks for Node.js, installs dependencies if needed, and opens the explorer in your browser.

### Options

```bash
whest visualizer --host 0.0.0.0 --port 8080   # bind to all interfaces on port 8080
whest visualizer --no-open                       # don't auto-open browser
```

On SSH/headless environments, the browser won't auto-open -- just follow the printed URL.

### Manual setup (fallback)

If `whest visualizer` doesn't work for your environment:

```bash
cd tools/whestbench-explorer
npm ci
npm run dev
```

Open `http://localhost:5173`.

## Expected outcome

You can interactively inspect network structure, layer behavior, and estimator comparisons.

## Suggested workflow

1. Start with small width/depth.
2. Vary seed to inspect structural changes.
3. Compare estimator behavior across layers.
4. Locate where errors concentrate.
5. Convert observations into Python estimator heuristics.

Official score semantics still come from:

```bash
whest run --estimator <path> --runner local
```

## Common first failure

Symptom: app does not start due to missing Node dependencies.

Fix: `whest visualizer` handles this automatically. For manual setup, run `npm ci` in `tools/whestbench-explorer` and retry `npm run dev`.

## Interpreting the visualization

The WhestBench Explorer shows neuron activations across layers:

- **Rows:** layers (top = first layer, bottom = output)
- **Columns:** neurons within each layer
- **Color intensity:** mean activation value

Patterns to look for:
- **Error at deep layers:** your method loses accuracy as correlations accumulate through layers
- **Sudden drops to zero:** ReLU is killing neuron groups — your variance estimates may be too narrow
- **Uniform predictions:** your estimator may not be using the weight structure

## Next step

- [Validate, Run, and Package](./validate-run-package.md)
- [Problem Setup](../concepts/problem-setup.md)

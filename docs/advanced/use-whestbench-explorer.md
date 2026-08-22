# Use WhestBench Explorer

> [← Documentation](../README.md)

![WhestBench Explorer -- a small network with 4 neurons and 5 layers, after running ground-truth estimation](../../assets/whestbench-explorer-visualization.svg)

## 🎯 When to use this page

When you want visual intuition about network behavior and where your estimator's error concentrates. The Explorer is optional: it is not the submission interface, and your leaderboard score never depends on it.

## 🚀 Open it

The Explorer is a separate, hosted React app:

| Where | URL |
|---|---|
| **Hosted (open in a browser)** | <https://aicrowd.github.io/whestbench-explorer/> |
| **Source / issues / PRs** | <https://github.com/AIcrowd/whestbench-explorer> |

Open the hosted URL, generate an MLP, propagate inputs, and inspect activations layer-by-layer. There's nothing to install.

> The Explorer was part of `whestbench` as a `whest visualizer` subcommand. As of whestbench commit `28c203f` (May 2026), it has its own repository with auto-deploy to GitHub Pages, and `whest visualizer` no longer exists.

## ✅ Expected outcome

An interactive view of network structure, layer behavior, and estimator-vs-ground-truth comparisons.

## Suggested workflow

1. Start with small width/depth.
2. Vary the seed to inspect how structure changes.
3. Compare estimator behavior across layers.
4. Locate where errors concentrate.
5. Convert observations into Python estimator heuristics, then verify with:
   ```bash
   uv run whest run --estimator estimator.py --runner local
   ```

The Explorer is for intuition. Official scoring still comes from `whest run`.

## Interpreting the visualization

The Explorer shows neuron activations across layers:

- **Rows:** layers (top = first layer, bottom = output)
- **Columns:** neurons within each layer
- **Color intensity:** mean activation value

Patterns to look for:

- **Error grows at deep layers:** your method loses accuracy as correlations accumulate through layers.
- **Sudden drops to zero:** ReLU is zeroing whole neuron groups; your variance estimates may be too narrow.
- **Uniform predictions:** your estimator may not be using the weight structure.

## ➡️ Next step

- [Validate, Run, and Package](../how-to/validate-run-package.md)
- [Problem Setup](../concepts/problem-setup.md)

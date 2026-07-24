# Advanced — Deeper tooling

> [← Documentation](../README.md)

You don't need either of these to ship a submission. They become useful once you're tuning aggressively and want better visibility into where FLOPs and time go.

| Doc | When to read |
|---|---|
| [profile-simulation.md](profile-simulation.md) | Benchmark the flopscope backend's correctness and wall-clock scaling across network sizes. |
| [use-whestbench-explorer.md](use-whestbench-explorer.md) | Open the hosted WhestBench Explorer (now an external app at [aicrowd.github.io/whestbench-explorer](https://aicrowd.github.io/whestbench-explorer/)) to inspect MLPs and ground-truth activations layer-by-layer. |

## ➡️ Where to look next

- Need a faster local iteration loop instead? → [How-to: use evaluation datasets](../how-to/use-evaluation-datasets.md).
- Looking for the FLOP cost of specific ops? → [Reference: flopscope primer](../reference/flopscope-primer.md).

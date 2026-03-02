# Problem Setup

## When To Use This Page

Use this page to understand what the estimator is predicting and why it is streamed by depth.

## Core Objects

Each evaluation call provides:

- one `Circuit` with `n` wires and `d` layers,
- one integer `budget`.

Your estimator must emit exactly `d` vectors, each with shape `(n,)`.

Row `i` represents your estimate of expected wire values after layer `i`.

## Ground Truth

Ground truth is approximated by Monte Carlo simulation over random inputs.
The evaluator computes empirical means by depth and by wire.

## Why Streaming Matters

The evaluator checks outputs and runtime at each depth, not only at the end.
This enforces both correctness of shape/count and budget-aware behavior throughout execution.

## Next

- [Scoring Model](./scoring-model.md)
- [Estimator Contract](../reference/estimator-contract.md)

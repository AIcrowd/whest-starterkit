# Problem Setup

## 🧠 When to use this page

Use this page when you want the technical framing of what is being estimated and why this benchmark uses streamed outputs.

## TL;DR

- Input: one random layered Boolean-circuit-style `Circuit` and one `budget`.
- Output: one `(n,)` prediction row per depth, for exactly `d` depths.
- Goal: estimate expected wire values under uniformly random inputs.

## What the estimator receives

Each evaluation call provides:

- one `Circuit` with `n` wires and `d` layers,
- one integer `budget`.

Your estimator must emit exactly `d` vectors, each with shape `(n,)`.

Row `i` is your estimate of expected wire values after layer `i`.

## Why this benchmark exists

This challenge is designed as a controlled setting for a broader research question: when can structure-aware estimation compete with or beat pure sampling under fixed compute?

ARC has publicly framed this direction as a key milestone for mechanistic estimation:

- [Competing with sampling](https://www.alignment.org/blog/competing-with-sampling/)
- [AlgZoo: uninterpreted models with fewer than 1,500 parameters](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)

This repository instantiates that question in random circuit families with explicit, reproducible local evaluation.

## Ground truth

Ground truth is approximated by Monte Carlo simulation over random inputs.
The evaluator computes empirical means by depth and wire.

## Why streaming matters

The evaluator checks both output validity and runtime behavior at each depth, not only at the end.

That is why your method must stream depth rows incrementally with `yield`.

## ➡️ Next step

- [Scoring Model](./scoring-model.md)
- [Inspect and Traverse Circuit Structure](../how-to/inspect-circuit-structure.md)
- [Estimator Contract](../reference/estimator-contract.md)

# Install and CLI Quickstart

## When to use this page

Use this page when setting up the starter kit from a fresh clone.

## Do this now

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the CLI in editable mode:

```bash
uv tool install -e .
```

Sanity-check CLI wiring:

```bash
whest smoke-test
```

Alternative invocation without global tool install:

```bash
uv run --with-editable . whest smoke-test
```

## ✅ Expected outcome

You can invoke `whest smoke-test`, see the built-in dashboard, and receive next-step commands for running your own estimator.

## 🛠 Common first failure

Symptom: `whest: command not found`

Fix: use the editable fallback invocation (`uv run --with-editable . whest smoke-test`) and confirm `uv` is on your `PATH`.

## ➡️ Next step

- [First Local Run](./first-local-run.md)
- [CLI Reference](../reference/cli-reference.md)
- [Whest Primer](../reference/whest-primer.md) — how FLOP tracking works under the hood

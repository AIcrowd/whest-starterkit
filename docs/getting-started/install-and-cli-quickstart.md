# Install and CLI Quickstart

## 🚀 When to use this page

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
cestim smoke-test
```

Alternative invocation without global tool install:

```bash
uv run --with-editable . cestim smoke-test
```

## ✅ Expected outcome

You can invoke `cestim smoke-test`, see the built-in dashboard, and receive next-step commands for running your own estimator.

## 🛠 Common first failure

Symptom: `cestim: command not found`

Fix: use the editable fallback invocation (`uv run --with-editable . cestim smoke-test`) and confirm `uv` is on your `PATH`.

## ➡️ Next step

- [First Local Run](./first-local-run.md)
- [CLI Reference](../reference/cli-reference.md)

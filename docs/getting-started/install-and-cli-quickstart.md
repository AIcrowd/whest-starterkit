# Install and CLI Quickstart

## When To Use This Page

Use this page when setting up the starter kit from a fresh clone.

## Prerequisites

- Python 3.10+
- shell access
- `curl` available

## Steps

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
cestim --json
```

Alternative invocation without global tool install:

```bash
uv run --with-editable . cestim --json
```

## Expected Outcome

You can invoke `cestim` locally and receive a valid JSON payload.

## Next

- [First Local Run](./first-local-run.md)
- [CLI Reference](../reference/cli-reference.md)

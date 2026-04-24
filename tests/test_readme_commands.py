"""Drift gate: every fenced bash block in README.md tagged ```bash-test must run cleanly.

Tag a fenced block with ```bash-test (instead of plain ```bash) to opt it into CI.
This excludes blocks like `git clone`, `gh release`, etc. that can't run in CI.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
FENCE_RE = re.compile(r"```bash-test\n(.*?)```", re.DOTALL)


def _readme_test_blocks() -> list[str]:
    if not README.exists():
        return []
    text = README.read_text()
    return [block.strip() for block in FENCE_RE.findall(text)]


@pytest.mark.parametrize("block", _readme_test_blocks())
def test_readme_bash_test_block_runs_cleanly(block: str):
    """Run each ```bash-test fenced block as a shell script. Non-zero exit = test fails."""
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", block],
        cwd=REPO_ROOT,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"README block failed:\n--- block ---\n{block}\n--- stdout ---\n"
        f"{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

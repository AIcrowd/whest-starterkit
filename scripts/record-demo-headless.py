#!/usr/bin/env python3
"""Headless demo-cast generator — no PTY required.

Companion to `make demo-cast` / `scripts/record-demo.sh`. `asciinema rec` needs a
PTY, which isn't available in some CI runners and agent sandboxes (`asciinema rec`
falls back to headless mode and then dies with `ENXIO: No such device`). This
script reproduces the *same* walkthrough without a PTY: it runs each command,
captures its real (colored) output, strips transient spinner frames, assembles an
asciicast-v3, and renders the GIF with `agg`.

The command sequence mirrors `scripts/record-demo.sh` — keep the two in sync.

Usage:
    python3 scripts/record-demo-headless.py              # capture fresh + build
    python3 scripts/record-demo-headless.py --reuse DIR  # rebuild from a prior capture
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLONE_URL = "https://github.com/AIcrowd/whest-starterkit.git"
DATASET = "hf://aicrowd/arc-whestbench-public-2026"

ESC = "\x1b"
GREEN = ESC + "[1;32m"
RESET = ESC + "[0m"
CSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# (displayed command, shell command to run | None, output filename | None, tail_lines | None)
# `None` run command = display-only step (e.g. `cd`). Mirrors scripts/record-demo.sh.
STEPS = [
    (f"git clone {CLONE_URL}", f"git clone --quiet {CLONE_URL} .", None, None),
    ("cd whest-starterkit", None, None, None),
    ("uv sync", "uv sync", "02_sync.out", 6),
    ("uv run python estimator.py", "uv run python estimator.py", "03_python.out", None),
    (
        "uv run whest validate --estimator estimator.py",
        "uv run whest validate --estimator estimator.py",
        "04_validate.out",
        None,
    ),
    (
        f"uv run whest run --estimator examples/03_covariance_propagation.py "
        f"--dataset {DATASET} --split mini --runner local",
        f"uv run whest run --estimator examples/03_covariance_propagation.py "
        f"--dataset {DATASET} --split mini --runner local",
        "05_run.out",
        None,
    ),
]

AGG_ARGS = ["--theme", "monokai", "--font-size", "14", "--last-frame-duration", "5",
            "--cols", "90", "--rows", "36"]


def _env() -> dict:
    e = dict(os.environ)
    e.update(TERM="xterm-256color", FORCE_COLOR="1", CLICOLOR_FORCE="1", COLUMNS="90")
    return e


def capture(capture_dir: Path) -> None:
    """Run the sequence in a fresh clone, writing each step's output to capture_dir."""
    capture_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp())
    # Step 1 clones into `work`; subsequent steps run there.
    rundir = work
    for _disp, cmd, outfile, tail in STEPS:
        if cmd is None:
            continue
        res = subprocess.run(cmd, shell=True, cwd=rundir, env=_env(),
                             capture_output=True, text=True)
        out = res.stdout + res.stderr
        if tail:
            out = "\n".join(out.splitlines()[-tail:]) + "\n"
        if outfile:
            (capture_dir / outfile).write_text(out, encoding="utf-8")
    # nothing else needed; clone left in `work` (a tempdir)


def clean(s: str) -> str:
    """Strip transient spinner/cursor sequences (keep SGR color + box-drawing)."""
    s = s.replace("\r", "")
    s = CSI.sub(lambda m: m.group(0) if m.group(0).endswith("m") else "", s)
    keep = []
    for line in s.split("\n"):
        plain = CSI.sub("", line)
        if "Importing estimator.py and running setup/predict checks" in plain:
            continue
        if any("⠀" <= ch <= "⣿" for ch in plain):  # braille spinner glyphs
            continue
        keep.append(line)
    return "\n".join(keep)


def _crlf(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\n", "\r\n")
    if s and not s.endswith("\r\n"):
        s += "\r\n"
    return s


def build_cast(capture_dir: Path, out_cast: Path) -> None:
    events: list = []
    events.append([0.006, "o", ESC + "c"])  # clear screen
    for i, (disp, _cmd, outfile, _tail) in enumerate(STEPS):
        events.append([0.5 if i == 0 else 1.0, "o", f"{GREEN}${RESET} {disp}\r\n"])
        if outfile:
            p = capture_dir / outfile
            out = _crlf(clean(p.read_text(encoding="utf-8", errors="replace"))) if p.exists() else ""
            if out:
                events.append([0.35, "o", out])
        events.append([0.15, "o", "\r\n"])
    events.append([2.0, "o", ""])  # hold final frame before the loop
    header = {"version": 3, "term": {"cols": 90, "rows": 36}, "idle_time_limit": 2.0,
              "title": "whest-starterkit: clone, sync, estimate, validate, score"}
    with out_cast.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def render_gif(out_cast: Path, out_gif: Path) -> None:
    subprocess.run(["agg", *AGG_ARGS, str(out_cast), str(out_gif)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reuse", metavar="DIR", help="rebuild from a prior capture dir (skip capture)")
    ap.add_argument("--out-cast", default=str(REPO / "assets" / "demo.cast"))
    ap.add_argument("--out-gif", default=str(REPO / "assets" / "demo.gif"))
    args = ap.parse_args()

    cap = Path(args.reuse) if args.reuse else Path(tempfile.mkdtemp())
    if not args.reuse:
        print(f"==> capturing demo output into {cap}")
        capture(cap)
    out_cast, out_gif = Path(args.out_cast), Path(args.out_gif)
    build_cast(cap, out_cast)
    print(f"==> wrote {out_cast}")
    render_gif(out_cast, out_gif)
    print(f"==> wrote {out_gif}")


if __name__ == "__main__":
    main()

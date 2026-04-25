#!/usr/bin/env bash
# Inner script for `make demo-cast`. Run via asciinema; takes a clean tempdir as $1.
set -e
cd "$1"

GREEN=$'\033[1;32m'
RESET=$'\033[0m'
prompt() { printf "%s\$%s %s\n" "$GREEN" "$RESET" "$1"; }

printf '\033c'

prompt "git clone https://github.com/AIcrowd/whest-starterkit.git"
git clone --quiet https://github.com/AIcrowd/whest-starterkit.git
echo

prompt "cd whest-starterkit"
cd whest-starterkit
echo

prompt "uv sync"
uv sync 2>&1 | tail -6
echo

prompt "uv run python estimator.py"
uv run python estimator.py
echo

prompt "uv run whest validate --estimator estimator.py"
uv run whest validate --estimator estimator.py
echo

prompt "uv run whest run --estimator examples/04_combined.py --runner local --n-mlps 20 --n-samples 100000 --seed 42"
uv run whest run --estimator examples/04_combined.py --runner local --n-mlps 20 --n-samples 100000 --seed 42

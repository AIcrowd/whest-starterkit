.PHONY: demo-cast install test lint help

help:
	@echo "Targets:"
	@echo "  install     uv sync --group dev"
	@echo "  test        run pytest"
	@echo "  lint        run ruff check"
	@echo "  demo-cast   record asciinema cast for README opener"

install:
	uv sync --group dev

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

# Records a 30-second cast of: clear, uv sync, python estimator.py
# Output: assets/demo.cast (committed) + assets/demo.svg (rendered)
# Requires: asciinema (brew install asciinema), agg or svg-term-cli for SVG.
demo-cast:
	@command -v asciinema >/dev/null || (echo "Install asciinema first: brew install asciinema"; exit 1)
	asciinema rec --overwrite --idle-time-limit=2 --title="whest-starterkit first 5 minutes" \
	  --command="bash -c 'clear; echo \$$ uv sync; uv sync --quiet; clear; echo \$$ python estimator.py; python estimator.py'" \
	  assets/demo.cast
	@command -v agg >/dev/null && agg assets/demo.cast assets/demo.svg || \
	  echo "Optional: install agg (cargo install --git https://github.com/asciinema/agg) to render SVG"

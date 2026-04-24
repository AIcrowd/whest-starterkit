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

# Records a fresh-clone walkthrough cast and renders to GIF.
# Output: assets/demo.cast (committed) + assets/demo.gif (committed).
# Requires: asciinema and agg (brew install asciinema agg).
demo-cast:
	@command -v asciinema >/dev/null || (echo "Install asciinema first: brew install asciinema"; exit 1)
	@command -v agg >/dev/null || (echo "Install agg first: brew install agg"; exit 1)
	@DEMO_DIR=$$(mktemp -d) && \
	  SCRIPT='cd '"$$DEMO_DIR"' && printf "\033c" && echo "$$ git clone https://github.com/AIcrowd/whest-starterkit.git" && git clone --quiet https://github.com/AIcrowd/whest-starterkit.git && echo && echo "$$ cd whest-starterkit" && cd whest-starterkit && echo && echo "$$ uv sync" && uv sync --quiet && echo && echo "$$ uv run python estimator.py" && uv run python estimator.py' && \
	  TERM=xterm-256color asciinema rec --overwrite --idle-time-limit=2 \
	    --title="whest-starterkit: first 5 minutes" \
	    --command="bash -c '$$SCRIPT'" assets/demo.cast && \
	  rm -rf "$$DEMO_DIR"
	agg --theme monokai --font-size 16 --last-frame-duration 5 \
	  --cols 80 --rows 18 assets/demo.cast assets/demo.gif

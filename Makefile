.PHONY: demo-cast demo-cast-headless install test lint help

help:
	@echo "Targets:"
	@echo "  install              uv sync --group dev"
	@echo "  test                 run pytest"
	@echo "  lint                 run ruff check"
	@echo "  demo-cast            record asciinema cast for README opener (needs a TTY)"
	@echo "  demo-cast-headless   build the cast without a TTY (CI / agent sandboxes)"

install:
	uv sync --group dev

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

# Records a fresh-clone walkthrough cast and renders to GIF.
# Sequence: git clone -> cd -> uv sync -> python estimator.py -> whest validate
# Output: assets/demo.cast (committed) + assets/demo.gif (committed).
# Requires: asciinema and agg (brew install asciinema agg).
demo-cast:
	@command -v asciinema >/dev/null || (echo "Install asciinema first: brew install asciinema"; exit 1)
	@command -v agg >/dev/null || (echo "Install agg first: brew install agg"; exit 1)
	@python3 -c "import os; os.close(os.openpty()[0])" 2>/dev/null || ( \
	  echo "No PTY available (CI / agent sandbox) — 'asciinema rec' needs a terminal."; \
	  echo "Run 'make demo-cast-headless' instead."; exit 1)
	@DEMO_DIR=$$(mktemp -d) && \
	  cp scripts/record-demo.sh "$$DEMO_DIR/demo.sh" && \
	  chmod +x "$$DEMO_DIR/demo.sh" && \
	  TERM=xterm-256color FORCE_COLOR=1 asciinema rec --overwrite --idle-time-limit=2 \
	    --window-size 90x36 \
	    --title="whest-starterkit: clone, sync, estimate, validate" \
	    --command="bash $$DEMO_DIR/demo.sh $$DEMO_DIR" assets/demo.cast && \
	  rm -rf "$$DEMO_DIR"
	agg --theme monokai --font-size 14 --last-frame-duration 5 \
	  --cols 90 --rows 36 assets/demo.cast assets/demo.gif

# Headless fallback for environments without a PTY (CI / agent sandboxes), where
# 'asciinema rec' can't run. Runs the same sequence, captures real output, strips
# transient spinner frames, assembles assets/demo.cast + renders assets/demo.gif.
# Requires: agg. Keep the sequence in sync with scripts/record-demo.sh.
demo-cast-headless:
	@command -v agg >/dev/null || (echo "Install agg first: brew install agg"; exit 1)
	python3 scripts/record-demo-headless.py

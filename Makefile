.PHONY: uv cuda cpu

# Install uv if not already installed
uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "uv not found. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "uv already installed."; \
	fi

# Run uv sync
cuda:
	source $HOME/.local/bin/env
	uv sync --extra dev --extra cuda

cpu:
	source $HOME/.local/bin/env
	uv sync --extra dev --extra metal
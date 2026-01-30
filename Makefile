.PHONY: uv cuda cpu

hf:
	curl -LsSf https://hf.co/cli/install.sh | bash

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
	uv sync --extra dev --extra cuda

cpu:
	uv sync --extra dev --extra metal
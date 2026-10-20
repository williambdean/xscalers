.PHONY: help docs serve test lint format build

.DEFAULT_GOAL := help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

serve: ## Serve the docs locally with live reload (http://localhost:8000)
	uv run zensical serve

build: ## Build the docs site into site/
	uv run zensical build

test: ## Run the test suite
	uv run pytest -q

lint: ## Run ruff and mypy
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

format: ## Apply ruff formatting
	uv run ruff format src tests

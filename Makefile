# Convenience wrappers. Every target is a thin shell-out to the venv/CLI so
# there is exactly one source of truth for behaviour.
PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip

.PHONY: help setup install test lint ingest ask eval stats serve serve-reload clean docker-up docker-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install everything (dev + model deps)
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

install: ## Install into an existing venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

test: ## Run the test suite
	$(PYTHON) -m pytest -q

ingest: ## Build/refresh the index from RAGQA_CORPUS_DIR
	$(PYTHON) cli.py ingest

ask: ## Ask a question: make ask Q="What is the retention period?"
	$(PYTHON) cli.py ask "$(Q)"

retrieve: ## Inspect retrieval only (no LLM): make retrieve Q="..."
	$(PYTHON) cli.py retrieve "$(Q)"

eval: ## Run the evaluation harness
	$(PYTHON) cli.py eval

stats: ## Show index statistics
	$(PYTHON) cli.py stats

config: ## Print the resolved configuration
	$(PYTHON) cli.py config

serve: ## Run the API server
	$(PYTHON) cli.py serve

serve-reload: ## Run the API server with auto-reload (dev)
	$(PYTHON) cli.py serve --reload

clean: ## Remove the index and caches (keeps the corpus and venv)
	rm -f data/index.sqlite3 data/index.sqlite3-*
	rm -rf .pytest_cache **/__pycache__

docker-up: ## Build and start the compose stack (api + ollama)
	docker compose up --build -d

docker-down: ## Stop the compose stack
	docker compose down

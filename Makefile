# AEROTWIN Makefile
# Usage: make <target>

PYTHON := python
PIP    := pip
PYTEST := pytest
UVICORN := uvicorn

.DEFAULT_GOAL := help

.PHONY: help setup dev test sim eval lint typecheck clean docker-up docker-down

help:
	@echo "AEROTWIN — Available targets:"
	@echo "  make setup        Install all dependencies (Python + Node)"
	@echo "  make dev          Start backend dev server (hot-reload)"
	@echo "  make test         Run all tests"
	@echo "  make sim          Run a simulated mission"
	@echo "  make eval         Run evaluation and generate report"
	@echo "  make lint         Run ruff linter"
	@echo "  make typecheck    Run mypy type-check"
	@echo "  make docker-up    Start Docker Compose stack"
	@echo "  make docker-down  Stop Docker Compose stack"
	@echo "  make clean        Remove generated files"

setup:
	@echo ">>> Installing Python dependencies..."
	$(PIP) install -r requirements.txt
	@echo ">>> Copying .env.example to .env (if not exists)..."
	@if not exist .env copy .env.example .env
	@echo ">>> Installing Node.js dependencies for web dashboard..."
	cd web && npm install
	@echo ">>> Setup complete. Run 'make dev' to start."

dev:
	@echo ">>> Starting AEROTWIN backend (hot-reload)..."
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	@echo ">>> Running all tests..."
	$(PYTEST) tests/ -v --tb=short

sim:
	@echo ">>> Running simulated mission..."
	$(PYTHON) -m core.simulator.run $(ARGS)

eval:
	@echo ">>> Running evaluation..."
	$(PYTHON) -m eval.run_eval
	@echo ">>> Evaluation report: eval/report.md"

lint:
	ruff check .

typecheck:
	mypy aerotwin --ignore-missing-imports

docker-up:
	docker compose up -d
	@echo ">>> Services up. Backend: http://localhost:8000 | Frontend: http://localhost:3000"

docker-down:
	docker compose down

clean:
	@echo ">>> Cleaning generated files..."
	if exist data\datasets rmdir /s /q data\datasets
	if exist data\aerotwin.db del /q data\aerotwin.db
	if exist eval\report.md del /q eval\report.md
	find . -type d -name __pycache__ -exec rm -rf {} + 2>nul || true
	find . -name "*.pyc" -delete 2>nul || true
	@echo ">>> Clean complete."

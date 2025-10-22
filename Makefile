.PHONY: help install format lint test test-unit test-integration test-cov clean check all

# Default target
help:
	@echo "ARIS - Sonar File Processing Toolkit"
	@echo ""
	@echo "Available commands:"
	@echo "  make install              Install dependencies with uv"
	@echo "  make format               Format code with ruff"
	@echo "  make lint                 Lint code with ruff"
	@echo "  make test                 Run all tests with pytest"
	@echo "  make test-unit            Run unit tests only"
	@echo "  make test-integration     Run integration tests only"
	@echo "  make test-cov             Run tests and generate coverage report"
	@echo "  make check                Run format check, lint, and tests"
	@echo "  make clean                Remove generated files and caches"
	@echo "  make all                  Run check target (format check + lint + test)"

# Install dependencies
install:
	uv sync

# Format code with ruff
format:
	uv run ruff format src/ scripts/ tests/

# Check formatting without making changes
format-check:
	uv run ruff format --check src/ scripts/ tests/

# Lint code with ruff
lint:
	uv run ruff check src/ scripts/ tests/

# Lint and automatically fix issues
lint-fix:
	uv run ruff check --fix src/ scripts/ tests/

# Run all tests
test:
	uv run pytest

# Run unit tests only
test-unit:
	uv run pytest tests/unit/

# Run integration tests only
test-integration:
	uv run pytest tests/integration/

# Run tests with coverage report
test-cov:
	uv run pytest --cov=src/aris --cov-report=term-missing --cov-report=html

# Run tests excluding slow tests
test-fast:
	uv run pytest -m "not slow"

# Check everything (format check, lint, test)
check: format-check lint test

# Run all quality checks and fixes
all: format lint test

# Clean generated files
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.orig" -delete

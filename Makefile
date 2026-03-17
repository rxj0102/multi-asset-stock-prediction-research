.PHONY: install install-dev lint format test test-cov notebook clean help

# ── Variables ─────────────────────────────────────────────────────────────────
PYTHON  := python3
PIP     := $(PYTHON) -m pip
PYTEST  := $(PYTHON) -m pytest
SRC     := src/stock_prediction
TESTS   := tests

# ── Installation ──────────────────────────────────────────────────────────────
install:          ## Install core dependencies
	$(PIP) install -e .

install-dev:      ## Install core + dev dependencies
	$(PIP) install -e ".[dev]"

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:             ## Lint with ruff
	$(PYTHON) -m ruff check $(SRC) $(TESTS)

format:           ## Auto-format with black
	$(PYTHON) -m black $(SRC) $(TESTS)

format-check:     ## Check formatting without applying changes
	$(PYTHON) -m black --check $(SRC) $(TESTS)

# ── Testing ───────────────────────────────────────────────────────────────────
test:             ## Run test suite
	$(PYTEST) $(TESTS)

test-cov:         ## Run tests with coverage report
	$(PYTEST) $(TESTS) --cov=$(SRC) --cov-report=term-missing --cov-report=html

# ── Jupyter ───────────────────────────────────────────────────────────────────
notebook:         ## Launch Jupyter notebook server
	jupyter notebook notebooks/

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean:            ## Remove build artefacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"     -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".coverage" -delete

help:             ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

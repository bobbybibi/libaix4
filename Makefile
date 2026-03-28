.PHONY: install run test lint format check clean all help serve train-knowledge

## Install dependencies
install:
	pip install -r requirements.txt

## Train the XOR neural network
run: install
	python train.py

## Run the test suite
test: install
	python -m pytest tests/ -v

## Lint code with ruff
lint:
	python -m ruff check .

## Auto-format code with ruff
format:
	python -m ruff format .

## Run lint + tests
check: lint test

## Full pipeline: install → lint → test → train
all: install lint test run

## Start the web server (port 5000)
serve: install
	python start.py

## Train the knowledge AI model
train-knowledge: install
	python train_knowledge.py

## Remove cached files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

## Show available targets
help:
	@echo ""
	@echo "Available targets:"
	@echo "  make install          Install dependencies"
	@echo "  make run              Train the XOR neural network"
	@echo "  make test             Run the pytest test suite"
	@echo "  make lint             Lint code with ruff"
	@echo "  make format           Auto-format code with ruff"
	@echo "  make check            Run lint + tests"
	@echo "  make all              Full pipeline (install → lint → test → train)"
	@echo "  make serve            Start the web server"
	@echo "  make train-knowledge  Train the knowledge AI model"
	@echo "  make clean            Remove build artifacts"
	@echo "  make help             Show this message"
	@echo ""

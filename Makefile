.PHONY: help install test lint format backtest notebook clean env-info

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies"
	@echo "  make test      - Run tests with coverage"
	@echo "  make lint      - Lint code (flake8 + mypy)"
	@echo "  make format    - Format code (black)"
	@echo "  make backtest  - Run default backtest"
	@echo "  make notebook  - Launch Jupyter"
	@echo "  make env-info  - Display environment information"
	@echo "  make clean     - Remove cache/logs"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ --line-length=100

backtest:
	python -m src.backtest.engine --config config/config.yaml

notebook:
	jupyter notebook notebooks/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage
	rm -rf logs/*.log

env-info:
	python -m src.utils.environment

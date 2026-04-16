.PHONY: install test lint clean

install:
	pip install -r requirements.txt

test:
	pytest

lint:
	ruff check src tests

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

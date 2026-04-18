IMAGE_NAME := credit-risk-api
IMAGE_TAG := latest

.PHONY: install test lint clean docker-build docker-run docker-stop

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

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:
	docker compose up -d

docker-stop:
	docker compose down

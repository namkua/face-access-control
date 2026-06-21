# Face Recognition MLOps - Makefile

.PHONY: help install test lint format clean docker-build docker-up docker-down deploy-local deploy-gke

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@egrep '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=api --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/ -v -m integration

test-load: ## Run load tests
	locust -f tests/load/load_test.py --host=http://localhost:8000 --headless -u 100 -r 10 -t 2m --html=load-test-report.html

lint: ## Run linting
	flake8 api/ --max-line-length=120
	mypy api/

format: ## Format code
	black api/ tests/
	isort api/ tests/

clean: ## Clean temporary files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '.coverage' -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start Docker services
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-restart: ## Restart Docker services
	docker-compose restart

generate-data: ## Generate synthetic data
	python scripts/data_generator.py

monitor: ## Open monitoring dashboards
	@echo "Opening monitoring dashboards..."
	@open http://localhost:3000 || xdg-open http://localhost:3000 || echo "Please open http://localhost:3000"
	@open http://localhost:9090 || xdg-open http://localhost:9090 || echo "Please open http://localhost:9090"


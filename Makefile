.PHONY: help install test lint format validate train check-performance pre-commit docker-build docker-up docker-down clean

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
PYLINT := pylint

help: ## Afficher cette aide
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installer toutes les dépendances
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Dépendances installées"

test: ## Exécuter les tests
	$(PYTEST) tests/ -v
	@echo "✓ Tests terminés"

lint: ## Vérifier la qualité du code (linting)
	@echo "Running Flake8..."
	$(FLAKE8) . --count --select=E9,F63,F7,F82 --show-source --statistics
	$(FLAKE8) . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "✓ Linting terminé"

format: ## Formater le code avec Black
	$(BLACK) .
	@echo "✓ Code formaté"

format-check: ## Vérifier le formatage sans modifier
	$(BLACK) --check --diff .

validate: ## Valider les données
	$(PYTHON) scripts/validate_data.py
	@echo "✓ Validation des données terminée"

train: ## Entraîner le modèle
	$(PYTHON) scripts/train_and_validate.py
	@echo "✓ Entraînement terminé"

check-performance: ## Vérifier les performances du modèle
	$(PYTHON) scripts/check_model_performance.py
	@echo "✓ Vérification des performances terminée"

pre-commit: ## Exécuter toutes les vérifications pré-commit
	$(PYTHON) scripts/pre_commit_check.py

ci-local: format-check lint validate train check-performance test ## Simuler le pipeline CI localement
	@echo ""
	@echo "✓ Pipeline CI local terminé avec succès!"

docker-build: ## Construire les images Docker
	docker build -t mlops-api:latest -f docker/Dockerfile.api .
	docker build -t mlops-train:latest -f docker/Dockerfile.train .
	@echo "✓ Images Docker construites"

docker-up: ## Démarrer les services Docker
	docker-compose up -d
	@echo "✓ Services démarrés"
	@echo "API disponible sur: http://localhost:8000"
	@echo "Docs disponible sur: http://localhost:8000/docs"

docker-down: ## Arrêter les services Docker
	docker-compose down
	@echo "✓ Services arrêtés"

docker-logs: ## Afficher les logs Docker
	docker-compose logs -f

clean: ## Nettoyer les fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ 2>/dev/null || true
	@echo "✓ Nettoyage terminé"

tag: ## Créer un tag de version (usage: make tag VERSION=v1.0.0)
	@if [ -z "$(VERSION)" ]; then \
		echo "Erreur: VERSION non définie. Usage: make tag VERSION=v1.0.0"; \
		exit 1; \
	fi
	git tag -a $(VERSION) -m "Release $(VERSION)"
	git push origin $(VERSION)
	@echo "✓ Tag $(VERSION) créé et poussé"

status: ## Afficher le statut du projet
	@echo "=== Git Status ==="
	@git status -s
	@echo ""
	@echo "=== Last Commit ==="
	@git log -1 --oneline
	@echo ""
	@echo "=== Current Version ==="
	@git describe --tags --abbrev=0 2>/dev/null || echo "No tags yet"

dev: install ## Setup environnement de développement
	@echo "✓ Environnement de développement configuré"
	@echo ""
	@echo "Pour commencer:"
	@echo "  - make validate    # Valider les données"
	@echo "  - make train       # Entraîner le modèle"
	@echo "  - make test        # Lancer les tests"
	@echo "  - make docker-up   # Démarrer l'API"

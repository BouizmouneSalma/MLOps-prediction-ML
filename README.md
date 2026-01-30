# 🏥 MLOps Diabetes Prediction System

[![CI Pipeline](https://img.shields.io/badge/CI-Pipeline-brightgreen)](https://github.com)
[![CD Pipeline](https://img.shields.io/badge/CD-Pipeline-blue)](https://github.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB)](https://www.python.org/)

## 📋 Table des Matières

- [À Propos](#à-propos)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Pipeline CI/CD](#pipeline-cicd)
- [Documentation](#documentation)
- [Structure du Projet](#structure-du-projet)
## 🎯 À Propos

Système MLOps complet pour la prédiction du diabète avec pipeline CI/CD automatisé, intégration MLflow pour le suivi des expériences, et déploiement continu via GitHub Actions.

### Objectifs du Projet

- 🤖 **Machine Learning** : Entraînement et comparaison de multiples modèles de classification
- 📊 **MLflow Tracking** : Suivi complet des expériences et versioning des modèles
- 🔄 **CI/CD** : Pipeline automatisé d'intégration et déploiement continu
- 🐳 **Containerisation** : Images Docker pour l'API et l'entraînement
- 🚀 **API REST** : Interface FastAPI pour les prédictions en temps réel
- 📈 **Monitoring** : Surveillance continue des performances du modèle

## ✨ Fonctionnalités

### Machine Learning

- ✅ **Multiples Algorithmes** : Random Forest, Gradient Boosting, XGBoost, etc.
- ✅ **GridSearch** : Optimisation automatique des hyperparamètres
- ✅ **Validation des Données** : Vérification automatique de la qualité des données
- ✅ **Métriques Complètes** : Accuracy, F1-Score, Precision, Recall, ROC-AUC
- ✅ **Visualisations** : Courbes ROC, matrices de confusion

### MLflow Integration

- ✅ **Tracking** : Logging automatique des paramètres, métriques et artefacts
- ✅ **Model Registry** : Gestion des versions de modèles
- ✅ **Staging → Production** : Promotion automatique des meilleurs modèles
- ✅ **Comparaison de Runs** : Interface web pour comparer les expériences

### API REST (FastAPI)

- ✅ **Prédiction en temps réel** : Endpoint `/predict` pour les prédictions
- ✅ **Validation Pydantic** : Validation automatique des données d'entrée
- ✅ **Documentation Interactive** : Swagger UI et ReDoc intégrés
- ✅ **Health Check** : Endpoint `/health` pour le monitoring
- ✅ **Model Info** : Endpoint `/model/info` pour les métadonnées du modèle

### CI/CD Pipeline

- ✅ **Tests Automatiques** : Validation du code, données et modèle à chaque push
- ✅ **Quality Checks** : Black, Flake8, Pylint pour la qualité du code
- ✅ **Build Docker** : Construction et versioning automatique des images
- ✅ **Déploiement Continu** : Staging et Production avec validation
- ✅ **Monitoring** : Alertes automatiques en cas de dégradation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                         │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   Code   │  │  Models  │  │   API    │  │  Tests   │       │
│  │ Training │  │   Logic  │  │ FastAPI  │  │  Pytest  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GitHub Actions (CI/CD)                      │
│                                                                  │
│  CI Pipeline                    CD Pipeline                      │
│  ├─ Code Quality               ├─ Build Docker Images           │
│  ├─ Data Validation            ├─ Push to Registry              │
│  ├─ Model Training             ├─ Deploy to Staging             │
│  ├─ Performance Check          └─ Deploy to Production          │
│  └─ API Tests                                                    │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
         ┌──────────────────┐  ┌──────────────────┐
         │   MLflow Server  │  │  Docker Registry │
         │                  │  │                  │
         │  - Experiments   │  │  - API Image     │
         │  - Model Registry│  │  - Train Image   │
         │  - Artifacts     │  │                  │
         └──────────────────┘  └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Production API  │
         │   (FastAPI)      │
         │                  │
         │  /predict        │
         │  /health         │
         │  /docs           │
         └──────────────────┘
```

## 🚀 Installation

### Prérequis

- Python 3.9+
- Git
- Docker (optionnel, pour le déploiement)
- Compte GitHub (pour CI/CD)

### Installation Locale

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/MLOps-prediction-ML.git
cd MLOps-prediction-ML

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Activer l'environnement (Linux/Mac)
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration MLflow

```bash
# Démarrer le serveur MLflow (optionnel pour local)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Accéder à l'interface MLflow
# http://localhost:5000
```

## 💻 Utilisation

### 1. Validation des Données

```bash
python scripts/validate_data.py
```

**Sortie attendue** : Rapport de validation avec statistiques sur les données

### 2. Entraînement du Modèle

```bash
# Entraînement complet avec MLflow tracking
python training/train.py

# Ou via le script de validation
python scripts/train_and_validate.py
```

**Le script va** :
- Charger et valider les données
- Entraîner plusieurs modèles
- Logger les résultats dans MLflow
- Sauvegarder le meilleur modèle
- Générer des visualisations

### 3. Lancer l'API

```bash
# Démarrer l'API FastAPI
cd api
uvicorn main:app --reload

# L'API sera disponible sur http://localhost:8000
```

### 4. Tester l'API

#### Via l'interface Swagger

Ouvrez votre navigateur : `http://localhost:8000/docs`

#### Via cURL

```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
  }'
```

#### Via Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
}

response = requests.post(url, json=data)
print(response.json())
```

### 5. Déploiement avec Docker

```bash
# Build de l'image API
docker build -t mlops-api:latest -f docker/Dockerfile.api .

# Lancer le conteneur
docker run -p 8000:8000 mlops-api:latest

# Ou utiliser docker-compose
docker-compose up -d
```

## 🔄 Pipeline CI/CD

### Déclenchement Automatique

Le pipeline CI/CD se déclenche automatiquement à chaque :
- Push sur `main` ou `develop`
- Pull Request vers `main`
- Création d'un tag `v*`

### Pipeline CI (Intégration Continue)

```yaml
1. Code Quality Check
   ├─ Black (formatage)
   ├─ Flake8 (linting)
   └─ Pylint (analyse statique)

2. Data Validation
   ├─ Vérification de l'intégrité
   ├─ Détection des valeurs manquantes
   └─ Génération du rapport

3. Model Training & Validation
   ├─ Entraînement avec MLflow
   ├─ Logging des métriques
   └─ Validation des performances

4. API Tests
   └─ Tests unitaires Pytest
```

### Pipeline CD (Déploiement Continu)

```yaml
1. Build Docker Images
   ├─ Image API
   └─ Image Training

2. Push to GitHub Container Registry
   ├─ Tag: latest
   ├─ Tag: {branch}-{sha}
   └─ Tag: v{version}

3. Deploy to Staging
   └─ Health check

4. Deploy to Production
   ├─ Validation manuelle (optionnel)
   └─ Health check
```

### Tests Locaux Avant Push

```bash
# Test rapide
.\test_pipeline.bat  # Windows
./test_pipeline.sh   # Linux/Mac

# Tests complets
python scripts/pre_commit_check.py
```

### Configuration GitHub Actions

Pour activer le CI/CD :

1. **Settings** → **Actions** → **General**
2. Cochez **"Read and write permissions"**
3. Cochez **"Allow GitHub Actions to create and approve pull requests"**
4. Cliquez **Save**

Voir [TESTING_GUIDE.md](TESTING_GUIDE.md) pour plus de détails.

## 📚 Documentation

Le projet inclut une documentation complète :

- **[START_HERE.md](START_HERE.md)** - 🎯 Guide de démarrage complet
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Guide de test et validation
- **[CI_CD_README.md](CI_CD_README.md)** - Documentation détaillée du CI/CD
- **[QUICKSTART_GUIDE.md](QUICKSTART_GUIDE.md)** - Guide de démarrage rapide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Récapitulatif technique

## 📁 Structure du Projet

```
MLOps-prediction-ML/
├── .github/
│   └── workflows/              # Workflows GitHub Actions
│       ├── ci.yml              # Pipeline d'intégration continue
│       ├── cd.yml              # Pipeline de déploiement continu
│       ├── release.yml         # Gestion des releases
│       └── monitoring.yml      # Monitoring des performances
│
├── api/                        # API FastAPI
│   ├── main.py                # Point d'entrée de l'API
│   └── model_loader.py        # Chargement du modèle depuis MLflow
│
├── training/                   # Scripts d'entraînement
│   ├── train.py               # Script principal d'entraînement
│   ├── models.py              # Définitions des modèles
│   └── data_validation.py     # Validation des données
│
├── scripts/                    # Scripts utilitaires
│   ├── validate_data.py       # Validation des données
│   ├── train_and_validate.py  # Entraînement + validation
│   ├── check_model_performance.py  # Vérification des seuils
│   ├── pre_commit_check.py    # Vérifications pré-commit
│   └── deploy_helper.py       # Helper de déploiement
│
├── tests/                      # Tests unitaires
│   └── test_api.py            # Tests de l'API
│
├── docker/                     # Dockerfiles
│   ├── Dockerfile.api         # Image pour l'API
│   └── Dockerfile.train       # Image pour l'entraînement
│
├── data/                       # Données
│   └── data.csv               # Dataset de diabète
│
├── reports/                    # Rapports générés
│   ├── data_validation_report.json
│   └── model_metrics.json
│
├── monitoring/                 # Configuration monitoring
│   └── prometheus.yml         # Configuration Prometheus
│
├── mlruns/                     # Artefacts MLflow (généré)
│
├── requirements.txt            # Dépendances Python
├── docker-compose.yml         # Configuration Docker Compose
├── Makefile                   # Commandes simplifiées
├── setup.cfg                  # Configuration linters
├── pyproject.toml             # Configuration Black
├── .env.example               # Template variables d'environnement
│
├── test_pipeline.bat          # Script de test (Windows)
├── test_pipeline.sh           # Script de test (Linux/Mac)
│
└── README.md                  # Ce fichier
```

```bash
# Tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_api.py -v

# Avec couverture
pytest tests/ --cov=. --cov-report=html
```

### Tests de l'API

```bash
# Démarrer l'API
uvicorn api.main:app --reload

## 🔧 Commandes Utiles

### Avec Makefile (Linux/Mac/WSL)

```bash
make help              # Afficher toutes les commandes
make install           # Installer les dépendances
make validate          # Valider les données
make train             # Entraîner le modèle
make test              # Lancer les tests
make lint              # Vérifier la qualité du code
make format            # Formater le code avec Black
make ci-local          # Simuler le CI localement
make docker-up         # Démarrer les services Docker
make docker-down       # Arrêter les services Docker
make clean             # Nettoyer les fichiers temporaires
```

### Sans Makefile (Windows)

```powershell
# Installation
pip install -r requirements.txt

# Validation et entraînement
python scripts/validate_data.py
python scripts/train_and_validate.py
python scripts/check_model_performance.py

# Tests
pytest tests/test_api.py -v

# Qualité du code
black .
flake8 .
pylint **/*.py

# Docker
docker-compose up -d
docker-compose down
```

## 📊 Métriques et Monitoring

### Seuils de Performance

Le modèle doit atteindre les seuils minimaux suivants :

- **Accuracy** : ≥ 70%
- **F1 Score** : ≥ 65%
- **Precision** : ≥ 60%
- **Recall** : ≥ 60%

Ces seuils peuvent être modifiés dans `scripts/check_model_performance.py`.

### Monitoring

- **Vérifications quotidiennes** : Workflow automatique chaque jour
- **Alertes** : Création automatique d'issues en cas de dégradation
- **Rapports** : Génération de rapports de performance

## 🐳 Docker

### Images Disponibles

Après le premier déploiement, les images Docker sont disponibles sur GitHub Container Registry :

```bash
# Récupérer les images
docker pull ghcr.io/VOTRE_USERNAME/mlops-prediction-ml-api:latest
docker pull ghcr.io/VOTRE_USERNAME/mlops-prediction-ml-train:latest

# Lancer l'API
docker run -p 8000:8000 ghcr.io/VOTRE_USERNAME/mlops-prediction-ml-api:latest
```

### Tags Disponibles

- `latest` : Dernière version de la branche principale
- `main-{sha}` : Version spécifique par commit
- `v{version}` : Version sémantique (ex: v1.0.0)

## 🤝 Contribution

### Workflow de Contribution

1. **Fork** le repository
2. **Créer une branche** : `git checkout -b feature/nouvelle-fonctionnalite`
3. **Développer** et tester localement
4. **Commit** : `git commit -m "feat: description de la fonctionnalité"`
5. **Push** : `git push origin feature/nouvelle-fonctionnalite`
6. **Créer une Pull Request**

Le pipeline CI validera automatiquement vos modifications.

### Standards de Code

- **Formatage** : Black (line-length=127)
- **Linting** : Flake8
- **Analyse statique** : Pylint
- **Tests** : Pytest avec couverture > 80%
- **Commits** : Convention Conventional Commits

## 🔐 Sécurité

- ✅ Validation stricte des entrées API (Pydantic)
- ✅ Pas de credentials dans le code
- ✅ Variables d'environnement pour la configuration
- ✅ Docker images scannées
- ✅ Dépendances régulièrement mises à jour

## 📝 Changelog

### v1.0.0 (2026-01-30)

- ✅ Pipeline CI/CD complet avec GitHub Actions
- ✅ Intégration MLflow pour le tracking
- ✅ API FastAPI avec validation Pydantic
- ✅ Images Docker pour API et Training
- ✅ Tests automatisés
- ✅ Monitoring des performances
- ✅ Documentation complète

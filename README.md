# Projet Météo Paris - ML Pipeline

Un projet de machine learning pour la prédiction météorologique à Paris avec une interface Streamlit, des pipelines Airflow et une architecture Docker.

## 🚀 Vue d'ensemble

Ce projet implémente un pipeline complet de machine learning pour prédire les conditions météorologiques à Paris. Il inclut :

- **Modèle ML** : Classification des conditions météo (XGBoost)
- **Interface web** : Application Streamlit avec carte interactive
- **Orchestration** : Pipelines Airflow pour l'entraînement et la prédiction
- **MLOps** : Intégration MLflow pour le suivi des expériences
- **Tests** : Suite de tests automatisés avec pytest

## 📁 Structure du projet

```
├── app/                    # Scripts Python principaux
├── airflow/               # DAGs Airflow
├── streamlit/             # Application web Streamlit
├── mlflow/                # Configuration MLflow
├── data/                  # Données météo
├── tests/                 # Tests unitaires
├── requirements.txt       # Dépendances Python
├── Dockerfile            # Configuration Docker
└── Jenkinsfile           # Pipeline CI/CD
```

## 📊 Données

Le projet utilise le fichier `data/weather_paris.csv` contenant :
- Température, humidité, pression
- Conditions météorologiques
- Données historiques de Paris

## 🛠 Technologies utilisées

- **ML** : scikit-learn, XGBoost, pandas, numpy
- **Interface** : Streamlit, folium (cartes)
- **Orchestration** : Apache Airflow
- **MLOps** : MLflow
- **Cloud** : AWS S3, boto3
- **Tests** : pytest, great_expectations
- **DevOps** : Docker, Jenkins


## 🎯 Utilisation


### Upload vers S3
python app/data_to_s3.py


### 1. Entraînement du modèle

```bash
# Version sans MLflow
python app/paris_meteo_no_mlflow.py

# Version avec fusion des données
python app/paris_meteo_fusion.py
```


### 2. Tests
```bash
# Lancer tous les tests
pytest tests/

# Tests spécifiques
pytest tests/test_paris_meteo_no_mlflow.py
```

### 3. Pipelines Airflow

```bash
cd airflow
docker-compose up -d
```

Accédez à `http://localhost:8080` pour gérer les DAGs :


### 4. Airflow Dags
- `paris_meteo_ml_pipeline_dag` : Entraînement du modèle
- `real_time_weather_prediction_dag` : Prédictions temps réel


## 🌟 Fonctionnalités

- ✅ Prédiction des conditions météo (Clear, Clouds, Rain, etc.)
- ✅ Interface web interactive avec carte
- ✅ Pipelines automatisés d'entraînement
- ✅ Suivi des expériences ML
- ✅ Tests automatisés
- ✅ Déploiement Docker
- ✅ Intégration CI/CD

## 🔧 Configuration

1. **Variables d'environnement** : Créer un fichier `.env` avec vos clés API
2. **MLflow** : Configurer l'URI de tracking dans les scripts
3. **AWS** : Configurer les credentials pour S3

## 📝 Licence

Ce projet est sous licence MIT.
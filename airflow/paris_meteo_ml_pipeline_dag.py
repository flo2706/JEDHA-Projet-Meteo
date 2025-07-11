# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report)
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
from dotenv import load_dotenv

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from airflow.models import Variable


# Configuration par défaut du DAG
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Définition du DAG
dag = DAG(
    'paris_meteo_ml_pipeline',
    default_args=default_args,
    description='Pipeline ML pour la prédiction météo de Paris',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'weather', 'xgboost', 'mlflow'],
)

# Configuration des chemins et variables
DATA_PATH = '/opt/airflow/data'
MODEL_PATH = '/opt/airflow/models'
WEATHER_CSV_FILE = 'weather_paris.csv'
CREDENTIALS_FILE = '/opt/airflow/config/credentials.txt'


def load_aws_credentials():
    """Charge les credentials AWS depuis la connexion Airflow aws_default"""
    try:
        # Récupérer la connexion AWS configurée dans Airflow
        aws_conn = BaseHook.get_connection('aws_default')
        
        # Configurer les variables d'environnement AWS
        if aws_conn.login:
            os.environ['AWS_ACCESS_KEY_ID'] = aws_conn.login
        if aws_conn.password:
            os.environ['AWS_SECRET_ACCESS_KEY'] = aws_conn.password
        if aws_conn.extra_dejson.get('region_name'):
            os.environ['AWS_DEFAULT_REGION'] = aws_conn.extra_dejson.get('region_name')
            
        print("AWS credentials loaded successfully from Airflow connection 'aws_default'")
        return True
        
    except Exception as e:
        print(f"Error loading AWS credentials from Airflow connection: {str(e)}")
        # Fallback: essayer de charger depuis le fichier credentials.txt
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                for line in f:
                    if line.strip() and '=' in line:
                        key, value = line.strip().split(' = ', 1)
                        os.environ[key] = value
            print("AWS credentials loaded from fallback credentials file")
            return True
        except FileNotFoundError:
            print(f"Fallback credentials.txt file not found at {CREDENTIALS_FILE}")
            return False


def setup_mlflow():
    """Configure MLflow"""
    credentials_loaded = load_aws_credentials()
    if not credentials_loaded:
        print("Warning: Could not load AWS credentials. MLflow logging may fail.")
    
    # Récupérer l'URI MLflow depuis les variables Airflow
    try:
        mlflow_uri = Variable.get("MLFLOW_TRACKING_URI")
        print(f"Using MLflow URI from Airflow variable: {mlflow_uri}")
    except Exception as e:
        # Fallback vers l'adresse par défaut si la variable n'existe pas
        mlflow_uri = "https://flodussart-mlflowprojectlead.hf.space"
        print(f"MLflow URI variable not found, using default: {mlflow_uri}")
        print(f"Error: {str(e)}")
    
    os.environ["APP_URI"] = mlflow_uri
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Meteo")


def test_aws_connection(**context):
    """Tâche 0: Tester la connexion AWS et les credentials"""
    print("Testing AWS connection and credentials...")
    
    try:
        # Charger les credentials AWS
        credentials_loaded = load_aws_credentials()
        
        if not credentials_loaded:
            raise ValueError("Could not load AWS credentials")
        
        # Vérifier que les variables d'environnement sont définies
        required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            raise ValueError(f"Missing AWS environment variables: {missing_vars}")
        
        print("✓ AWS credentials loaded successfully")
        print(f"✓ AWS_ACCESS_KEY_ID: {os.environ.get('AWS_ACCESS_KEY_ID', 'Not set')[:10]}...")
        print(f"✓ AWS_SECRET_ACCESS_KEY: {'*' * 10}...")
        
        # Optionnel: Test basique de connexion AWS (si boto3 est disponible)
        try:
            import boto3
            # Test de création d'un client S3 (sans faire d'appel API réel)
            s3_client = boto3.client('s3')
            print("✓ boto3 S3 client created successfully")
        except ImportError:
            print("Note: boto3 not available for connection testing")
        except Exception as e:
            print(f"Warning: Could not create S3 client: {str(e)}")
            
        return True
        
    except Exception as e:
        print(f"❌ AWS connection test failed: {str(e)}")
        raise


def load_and_validate_data(**context):
    """Tâche 1: Charger et valider les données"""
    print("Loading and validating weather data...")
    
    # Chemin du fichier CSV
    csv_path = os.path.join(DATA_PATH, WEATHER_CSV_FILE)
    
    # Charger les données
    try:
        paris_df = pd.read_csv(csv_path)
        print(f"Data loaded successfully. Shape: {paris_df.shape}")
        
        # Validation basique
        if paris_df.empty:
            raise ValueError("Dataset is empty")
        
        required_columns = ['datetime', 'weather_main', 'weather_description']
        missing_columns = [col for col in required_columns if col not in paris_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Columns: {list(paris_df.columns)}")
        print(f"Weather main categories: {paris_df['weather_main'].unique()}")
        
        # Sauvegarder les données validées
        validated_data_path = f"{MODEL_PATH}/validated_data.pkl"
        os.makedirs(MODEL_PATH, exist_ok=True)
        paris_df.to_pickle(validated_data_path)
        
        return validated_data_path
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def preprocess_data(**context):
    """Tâche 2: Préprocessing et feature engineering"""
    print("Preprocessing data and engineering features...")
    
    # Récupérer le chemin des données validées
    validated_data_path = f"{MODEL_PATH}/validated_data.pkl"
    paris_df = pd.read_pickle(validated_data_path)
    
    # Fusionner les classes
    paris_df['weather_main'] = paris_df['weather_main'].replace({'Drizzle': 'Rain', 'Mist': 'Fog'})
    print(f"After merging classes: {paris_df['weather_main'].unique()}")
    
    # Vérifier la distribution des classes et filtrer celles avec très peu d'échantillons
    class_counts = paris_df['weather_main'].value_counts()
    print(f"Class distribution before filtering: {class_counts.to_dict()}")
    
    # Garder seulement les classes avec au moins 5 échantillons pour avoir suffisamment de données
    min_samples_per_class = 5
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    
    if len(valid_classes) < len(class_counts):
        print(f"Filtering out classes with less than {min_samples_per_class} samples")
        paris_df = paris_df[paris_df['weather_main'].isin(valid_classes)]
        print(f"Remaining classes: {paris_df['weather_main'].value_counts().to_dict()}")
    
    print(f"Final dataset shape after filtering: {paris_df.shape}")
    
    # Feature engineering temporel
    paris_df['datetime'] = pd.to_datetime(paris_df['datetime'])
    paris_df['hour'] = paris_df['datetime'].dt.hour
    paris_df['day'] = paris_df['datetime'].dt.day
    paris_df['month'] = paris_df['datetime'].dt.month
    paris_df['weekday'] = paris_df['datetime'].dt.weekday
    paris_df['is_weekend'] = paris_df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Features cycliques
    paris_df['hour_sin'] = np.sin(2 * np.pi * paris_df['hour'] / 24)
    paris_df['hour_cos'] = np.cos(2 * np.pi * paris_df['hour'] / 24)
    paris_df['month_sin'] = np.sin(2 * np.pi * paris_df['month'] / 12)
    paris_df['month_cos'] = np.cos(2 * np.pi * paris_df['month'] / 12)
    
    # Supprimer les colonnes non nécessaires
    paris_df = paris_df.drop(['datetime', 'weather_description'], axis=1)
    
    # Encodage de la variable cible
    le = LabelEncoder()
    paris_df['weather_main_encoded'] = le.fit_transform(paris_df['weather_main'])
    
    print(f"Preprocessed data shape: {paris_df.shape}")
    print(f"Target classes: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Sauvegarder les données préprocessées et l'encodeur
    preprocessed_data_path = f"{MODEL_PATH}/preprocessed_data.pkl"
    label_encoder_path = f"{MODEL_PATH}/label_encoder.pkl"
    
    paris_df.to_pickle(preprocessed_data_path)
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(le, f)
    
    return {
        'preprocessed_data_path': preprocessed_data_path,
        'label_encoder_path': label_encoder_path
    }


def split_data(**context):
    """Tâche 3: Division des données en train/test"""
    print("Splitting data into train and test sets...")
    
    # Charger les données préprocessées
    preprocessed_data_path = f"{MODEL_PATH}/preprocessed_data.pkl"
    paris_df = pd.read_pickle(preprocessed_data_path)
    
    # Séparer les features et la target
    X = paris_df.drop(columns=['weather_main', 'weather_main_encoded'])
    Y = paris_df['weather_main_encoded']
    
    # Vérifier la distribution des classes
    class_counts = Y.value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    # Vérifier si toutes les classes ont au moins 2 échantillons pour la stratification
    min_class_count = class_counts.min()
    use_stratify = min_class_count >= 2
    
    if use_stratify:
        print("Using stratified split")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )
    else:
        print(f"Cannot use stratified split (min class count: {min_class_count}). Using random split.")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
    
    print(f"Training set shape: X_train={X_train.shape}, Y_train={Y_train.shape}")
    print(f"Test set shape: X_test={X_test.shape}, Y_test={Y_test.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Sauvegarder les datasets
    train_test_paths = {
        'X_train_path': f"{MODEL_PATH}/X_train.pkl",
        'X_test_path': f"{MODEL_PATH}/X_test.pkl",
        'Y_train_path': f"{MODEL_PATH}/Y_train.pkl",
        'Y_test_path': f"{MODEL_PATH}/Y_test.pkl"
    }
    
    X_train.to_pickle(train_test_paths['X_train_path'])
    X_test.to_pickle(train_test_paths['X_test_path'])
    Y_train.to_pickle(train_test_paths['Y_train_path'])
    Y_test.to_pickle(train_test_paths['Y_test_path'])
    
    return train_test_paths


def train_model(**context):
    """Tâche 4: Entraînement du modèle XGBoost"""
    print("Training XGBoost model...")
    
    # Charger les données d'entraînement
    X_train = pd.read_pickle(f"{MODEL_PATH}/X_train.pkl")
    Y_train = pd.read_pickle(f"{MODEL_PATH}/Y_train.pkl")
    
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    
    # Initialiser et entraîner le modèle
    model = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=100
    )
    
    model.fit(X_train, Y_train)
    print("Model training completed")
    
    # Sauvegarder le modèle
    model_path = f"{MODEL_PATH}/xgboost_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path


def evaluate_model(**context):
    """Tâche 5: Évaluation du modèle et logging MLflow"""
    print("Evaluating model and logging to MLflow...")
    
    # Setup MLflow
    setup_mlflow()
    
    # Charger le modèle et les données
    with open(f"{MODEL_PATH}/xgboost_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    X_train = pd.read_pickle(f"{MODEL_PATH}/X_train.pkl")
    X_test = pd.read_pickle(f"{MODEL_PATH}/X_test.pkl")
    Y_train = pd.read_pickle(f"{MODEL_PATH}/Y_train.pkl")
    Y_test = pd.read_pickle(f"{MODEL_PATH}/Y_test.pkl")
    
    with open(f"{MODEL_PATH}/label_encoder.pkl", 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calcul des métriques
    metrics = {
        'accuracy_train': accuracy_score(Y_train, y_train_pred),
        'precision_train': precision_score(Y_train, y_train_pred, average='weighted'),
        'recall_train': recall_score(Y_train, y_train_pred, average='weighted'),
        'f1_train': f1_score(Y_train, y_train_pred, average='weighted'),
        'accuracy_test': accuracy_score(Y_test, y_test_pred),
        'precision_test': precision_score(Y_test, y_test_pred, average='weighted'),
        'recall_test': recall_score(Y_test, y_test_pred, average='weighted'),
        'f1_test': f1_score(Y_test, y_test_pred, average='weighted'),
    }
    
    print("Model Performance Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # MLflow logging
    experiment = mlflow.get_experiment_by_name("Meteo")
    run_name = f"Airflow_XGBClassifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
        # Log des paramètres du modèle
        mlflow.log_params({
            'model_type': 'XGBClassifier',
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'learning_rate': model.learning_rate,
            'n_features': X_train.shape[1],
            'n_classes': len(label_encoder.classes_)
        })
        
        # Log des métriques
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name.replace('_', ' ').title(), value)
        
        # Log des artefacts
        confusion_matrix_test = confusion_matrix(Y_test, y_test_pred)
        classification_report_test = classification_report(Y_test, y_test_pred)
        
        mlflow.log_text(str(confusion_matrix_test), "confusion_matrix_test.txt")
        mlflow.log_text(classification_report_test, "classification_report_test.txt")
        
        # Log du modèle
        mlflow.xgboost.log_model(model, "model")
        
        print(f"MLflow run completed: {run.info.run_id}")
    
    return metrics


# Définition des tâches
task_test_aws = PythonOperator(
    task_id='test_aws_connection',
    python_callable=test_aws_connection,
    dag=dag,
)

task_load_data = PythonOperator(
    task_id='load_and_validate_data',
    python_callable=load_and_validate_data,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

task_split = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

task_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# Définition des dépendances
task_test_aws >> task_load_data >> task_preprocess >> task_split >> task_train >> task_evaluate

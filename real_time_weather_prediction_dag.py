# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import pickle
import requests
import json
from sklearn.preprocessing import LabelEncoder
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
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

# Définition du DAG
dag = DAG(
    'real_time_weather_prediction',
    default_args=default_args,
    description='Prédiction météo en temps réel avec MLflow et OpenWeather API',
    schedule_interval=timedelta(hours=1),  # Exécution toutes les heures
    catchup=False,
    tags=['ml', 'weather', 'prediction', 'real-time', 'openweather', 'mlflow'],
)

# Configuration des chemins et variables
MODEL_PATH = '/opt/airflow/models'
PREDICTIONS_PATH = '/opt/airflow/predictions'
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/3.0/onecall"
API_KEY = "8021a55eaa75f382697bb1956b2589b4"
LATITUDE = 48.8566  # Paris latitude (correspond à l'API testée)
LONGITUDE = 2.3522  # Paris longitude (correspond à l'API testée)

# Mapping des conditions météo OpenWeather vers nos classes
WEATHER_MAPPING = {
    'Clear': 'Clear',
    'Clouds': 'Clouds',
    'Rain': 'Rain',
    'Drizzle': 'Rain',  # Fusionné avec Rain comme dans l'entraînement
    'Thunderstorm': 'Thunderstorm',
    'Snow': 'Snow',
    'Mist': 'Fog',      # Fusionné avec Fog comme dans l'entraînement
    'Fog': 'Fog',
    'Smoke': 'Fog',
    'Haze': 'Fog',
    'Dust': 'Fog',
    'Sand': 'Fog'
}


def safe_json_serialize(obj):
    """Convertit les objets non-sérialisables JSON en formats compatibles"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj


def load_aws_credentials():
    """Charge les credentials AWS depuis la connexion Airflow aws_default"""
    try:
        aws_conn = BaseHook.get_connection('aws_default')
        
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
        return False


def setup_mlflow():
    """Configure MLflow"""
    credentials_loaded = load_aws_credentials()
    if not credentials_loaded:
        print("Warning: Could not load AWS credentials. MLflow may fail.")
    
    # Récupérer l'URI MLflow depuis les variables Airflow
    try:
        mlflow_uri = Variable.get("mlflow_uri")
        print(f"Using MLflow URI from Airflow variable: {mlflow_uri}")
    except Exception as e:
        # Fallback vers l'adresse par défaut si la variable n'existe pas
        mlflow_uri = "https://f8fc-91-164-131-62.ngrok-free.app"
        print(f"MLflow URI variable not found, using default: {mlflow_uri}")
        print(f"Error: {str(e)}")
    
    os.environ["APP_URI"] = mlflow_uri
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Meteo")


def fetch_weather_data(**context):
    """Tâche 1: Récupérer les données météo en temps réel depuis OpenWeather API"""
    print("Fetching real-time weather data from OpenWeather API...")
    
    # Construire l'URL de l'API
    params = {
        'lat': LATITUDE,
        'lon': LONGITUDE,
        'appid': API_KEY,
        'units': 'metric',  # Températures en Celsius
        'exclude': 'minutely,alerts'  # Exclure les données non nécessaires
    }
    
    try:
        # Faire l'appel API
        response = requests.get(OPENWEATHER_API_URL, params=params, timeout=30)
        response.raise_for_status()
        
        weather_data = response.json()
        print(f"API Response received successfully")
        
        # Extraire les données actuelles
        current = weather_data['current']
        current_time = datetime.fromtimestamp(current['dt'])
        
        # Préparer les données pour la prédiction
        weather_features = {
            'datetime': current_time,
            'temp': current['temp'],
            'feels_like': current['feels_like'],
            'pressure': current['pressure'],
            'humidity': current['humidity'],
            'dew_point': current['dew_point'],
            'uvi': current['uvi'],
            'clouds': current['clouds'],
            'visibility': current.get('visibility', 10000),  # défaut 10km si absent
            'wind_speed': current['wind_speed'],
            'wind_deg': current.get('wind_deg', 0),  # défaut 0 si absent
            'weather_main': current['weather'][0]['main'],
            'weather_description': current['weather'][0]['description'],
            # Ajouter des champs supplémentaires si disponibles
            'sunrise': weather_data['current'].get('sunrise', 0),
            'sunset': weather_data['current'].get('sunset', 0)
        }
        
        # Ajouter les données de précipitations si disponibles
        if 'rain' in current:
            weather_features['rain_1h'] = current['rain'].get('1h', 0)
        else:
            weather_features['rain_1h'] = 0
            
        if 'snow' in current:
            weather_features['snow_1h'] = current['snow'].get('1h', 0)
        else:
            weather_features['snow_1h'] = 0
        
        print(f"Current weather in Paris: {weather_features['weather_main']} - {weather_features['weather_description']}")
        print(f"Temperature: {weather_features['temp']}°C")
        print(f"Humidity: {weather_features['humidity']}%")
        
        # Sauvegarder les données brutes
        os.makedirs(PREDICTIONS_PATH, exist_ok=True)
        raw_data_path = f"{PREDICTIONS_PATH}/raw_weather_data_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Préparer les données pour la sérialisation JSON (convertir datetime en string)
        weather_features_serializable = weather_features.copy()
        weather_features_serializable['datetime'] = current_time.isoformat()
        
        # Sérialiser de manière sécurisée toute la réponse API
        safe_weather_data = safe_json_serialize(weather_data)
        
        with open(raw_data_path, 'w') as f:
            json.dump({
                'timestamp': current_time.isoformat(),
                'weather_features': weather_features_serializable,
                'full_response': safe_weather_data
            }, f, indent=2)
        
        # Préparer les données de retour (toutes sérialisables pour XCom)
        return_data = {
            'raw_data_path': raw_data_path,
            'weather_features': weather_features_serializable,  # Utiliser la version sérialisable
            'timestamp': current_time.isoformat()
        }
        
        # S'assurer que tout est sérialisable pour XCom
        return safe_json_serialize(return_data)
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise


def preprocess_realtime_data(**context):
    """Tâche 2: Préprocesser les données en temps réel pour la prédiction"""
    print("Preprocessing real-time weather data...")
    
    # Récupérer les données de la tâche précédente
    ti = context['ti']
    weather_info = ti.xcom_pull(task_ids='fetch_weather_data')
    weather_features = weather_info['weather_features']
    
    # Créer un DataFrame avec les données actuelles
    df = pd.DataFrame([weather_features])
    
    # Appliquer le même preprocessing que durant l'entraînement
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Feature engineering temporel
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Features cycliques
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Mapper la condition météo actuelle vers nos classes d'entraînement
    current_weather = df['weather_main'].iloc[0]
    mapped_weather = WEATHER_MAPPING.get(current_weather, current_weather)
    df['weather_main_mapped'] = mapped_weather
    
    print(f"Original weather: {current_weather}, Mapped to: {mapped_weather}")
    
    # Supprimer les colonnes non nécessaires pour la prédiction
    # Garder seulement les features qui étaient disponibles durant l'entraînement
    columns_to_drop = [
        'datetime', 'weather_main', 'weather_description', 'weather_main_mapped',
        'sunrise', 'sunset',  # Supprimer les nouvelles colonnes non utilisées dans l'entraînement
        'snow_1h'  # Garder rain_1h mais supprimer snow_1h (moins commun)
    ]
    
    available_columns = [col for col in columns_to_drop if col in df.columns]
    feature_columns = [col for col in df.columns if col not in available_columns]
    
    print(f"Columns to drop: {[col for col in columns_to_drop if col in df.columns]}")
    print(f"Remaining feature columns: {feature_columns}")
    
    X_realtime = df[feature_columns]
    
    print(f"Preprocessed features shape: {X_realtime.shape}")
    print(f"Features: {list(X_realtime.columns)}")
    print(f"Sample of feature values:")
    for col in X_realtime.columns:
        print(f"  {col}: {X_realtime[col].iloc[0]}")
    
    # Vérifier que nous avons des valeurs numériques valides
    for col in X_realtime.columns:
        if X_realtime[col].iloc[0] is None or pd.isna(X_realtime[col].iloc[0]):
            print(f"Warning: Feature {col} has null value, setting to 0")
            X_realtime[col] = X_realtime[col].fillna(0)
    
    # Sauvegarder les données préprocessées
    preprocessed_path = f"{PREDICTIONS_PATH}/preprocessed_data_{weather_info['timestamp'].replace(':', '-')}.pkl"
    X_realtime.to_pickle(preprocessed_path)
    
    return {
        'preprocessed_path': preprocessed_path,
        'actual_weather': mapped_weather,
        'timestamp': weather_info['timestamp'],  # Déjà une string
        'original_features': safe_json_serialize(weather_features)  # S'assurer de la sérialisabilité
    }


def load_model_from_mlflow(**context):
    """Tâche 3: Charger le dernier modèle depuis MLflow"""
    print("Loading latest model from MLflow...")
    
    # Setup MLflow
    setup_mlflow()
    
    try:
        # Récupérer l'expérience
        experiment = mlflow.get_experiment_by_name("Meteo")
        
        if experiment is None:
            raise ValueError("MLflow experiment 'Meteo' not found")
        
        print(f"Found MLflow experiment: {experiment.name} (ID: {experiment.experiment_id})")
        
        # Chercher les runs (sans order_by pour éviter les problèmes de syntaxe MLflow)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=50  # Augmenter pour avoir plus de chances de trouver un modèle
        )
        
        print(f"Found {len(runs)} runs in experiment")
        
        if runs.empty:
            raise ValueError("No runs found in MLflow experiment 'Meteo'")
        
        # Trier manuellement par start_time si la colonne existe
        if 'start_time' in runs.columns:
            runs = runs.sort_values('start_time', ascending=False)
            print("Runs sorted by start_time (newest first)")
        else:
            print("Warning: start_time column not found, using original order")
        
        # Filtrer les runs terminés avec succès
        finished_runs = runs[runs['status'] == 'FINISHED'] if 'status' in runs.columns else runs
        
        if finished_runs.empty:
            print("No finished runs found, will try all available runs")
            finished_runs = runs
            
        print(f"Will try {len(finished_runs)} runs to find a model")
        
        # Trouver un run avec un modèle artifact
        model_loaded = False
        model = None
        label_encoder = None
        run_id = None
        
        print(f"Available columns in runs DataFrame: {list(finished_runs.columns)}")
        
        for idx, (_, run) in enumerate(finished_runs.iterrows()):
            try:
                run_id = run['run_id']
                run_status = run.get('status', 'Unknown')
                run_name = run.get('tags.mlflow.runName', 'Unknown')
                
                print(f"[{idx+1}/{len(finished_runs)}] Trying run: {run_id}")
                print(f"  Status: {run_status}, Name: {run_name}")
                
                # Charger le modèle
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.xgboost.load_model(model_uri)
                
                print(f"✓ Model loaded successfully from run: {run_id}")
                print(f"  Model type: {type(model)}")
                
                model_loaded = True
                break
                
            except Exception as e:
                print(f"  ❌ Could not load model: {str(e)}")
                continue
        
        if not model_loaded:
            # Fallback: essayer de charger depuis un fichier local si disponible
            print("Could not load any model from MLflow, trying local fallback...")
            try:
                local_model_path = f"{MODEL_PATH}/xgboost_model.pkl"
                if os.path.exists(local_model_path):
                    with open(local_model_path, 'rb') as f:
                        model = pickle.load(f)
                    print("✓ Fallback: Model loaded from local file")
                    run_id = "local_fallback"
                    model_loaded = True
                else:
                    raise ValueError("No local model file found either")
            except Exception as e:
                print(f"Fallback failed: {str(e)}")
                raise ValueError("Could not load any model from MLflow or local files")
        
        # Essayer de charger le label encoder depuis le modèle d'entraînement local
        # (fallback car il pourrait ne pas être dans MLflow)
        try:
            label_encoder_path = f"{MODEL_PATH}/label_encoder.pkl"
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            print("✓ Label encoder loaded from local file")
        except:
            print("Warning: Could not load label encoder from local file")
            # Créer un label encoder par défaut avec les classes connues
            label_encoder = LabelEncoder()
            known_classes = ['Clear', 'Clouds', 'Rain', 'Thunderstorm', 'Snow', 'Fog']
            label_encoder.fit(known_classes)
            print("✓ Default label encoder created with known classes")
         # Sauvegarder le modèle et le label encoder localement pour cette prédiction
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"{PREDICTIONS_PATH}/model_{timestamp}.pkl"
        label_encoder_path = f"{PREDICTIONS_PATH}/label_encoder_{timestamp}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        return {
            'model_path': model_path,
            'label_encoder_path': label_encoder_path,  # Chemin au lieu de l'objet
            'run_id': run_id,
            'timestamp': timestamp
        }
        
    except Exception as e:
        print(f"Error loading model from MLflow: {str(e)}")
        raise


def make_prediction(**context):
    """Tâche 4: Faire la prédiction avec le modèle chargé"""
    print("Making weather prediction...")
    
    # Récupérer les données des tâches précédentes
    ti = context['ti']
    preprocessed_info = ti.xcom_pull(task_ids='preprocess_realtime_data')
    model_info = ti.xcom_pull(task_ids='load_model_from_mlflow')
    
    # Charger les données préprocessées
    X_realtime = pd.read_pickle(preprocessed_info['preprocessed_path'])
    
    # Charger le modèle et le label encoder
    with open(model_info['model_path'], 'rb') as f:
        model = pickle.load(f)
    
    with open(model_info['label_encoder_path'], 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Model expects features: {getattr(model, 'feature_names_in_', 'Not available')}")
    print(f"Real-time data features: {list(X_realtime.columns)}")
    print(f"Real-time data shape: {X_realtime.shape}")
    print(f"Real-time data sample:")
    print(X_realtime.head())
    
    # Essayer de charger les features d'entraînement depuis le fichier local
    try:
        training_data_path = f"{MODEL_PATH}/X_train.pkl"
        if os.path.exists(training_data_path):
            X_train_sample = pd.read_pickle(training_data_path)
            expected_features = list(X_train_sample.columns)
            print(f"Expected features from training data: {expected_features}")
            print(f"Expected number of features: {len(expected_features)}")
        else:
            expected_features = None
            print("Training data not found locally")
    except Exception as e:
        expected_features = None
        print(f"Could not load training features: {str(e)}")
    
    try:
        # Vérifier la compatibilité des features
        current_features = X_realtime.columns.tolist()
        
        if expected_features is not None:
            # Utiliser les features d'entraînement comme référence
            missing_features = set(expected_features) - set(current_features)
            extra_features = set(current_features) - set(expected_features)
            
            print(f"Missing features: {missing_features}")
            print(f"Extra features: {extra_features}")
            
            if missing_features:
                print(f"Adding missing features with default values: {missing_features}")
                for feature in missing_features:
                    X_realtime[feature] = 0
            
            if extra_features:
                print(f"Removing extra features: {extra_features}")
            
            # Garder seulement les features d'entraînement dans le bon ordre
            X_realtime = X_realtime[expected_features]
            print(f"Features adjusted to match training data")
            
        elif hasattr(model, 'feature_names_in_'):
            # Fallback sur les features du modèle si disponibles
            expected_features = model.feature_names_in_
            missing_features = set(expected_features) - set(current_features)
            extra_features = set(current_features) - set(expected_features)
            
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                for feature in missing_features:
                    X_realtime[feature] = 0
                    print(f"Added missing feature '{feature}' with default value 0")
            
            if extra_features:
                print(f"Warning: Extra features will be ignored: {extra_features}")
            
            # Réorganiser les colonnes dans l'ordre attendu
            X_realtime = X_realtime[expected_features]
            print(f"Features reordered to match model expectations")
        else:
            print(f"Warning: Could not determine expected features. Model expects {model.num_features() if hasattr(model, 'num_features') else 'unknown'} features")
            print(f"Current data has {X_realtime.shape[1]} features")
            
            # Utiliser num_features() pour ajuster automatiquement
            if hasattr(model, 'num_features'):
                expected_num_features = model.num_features()
                current_num_features = X_realtime.shape[1]
                
                if current_num_features < expected_num_features:
                    # Ajouter des features manquantes avec valeur 0
                    features_to_add = expected_num_features - current_num_features
                    print(f"Adding {features_to_add} missing features with default value 0")
                    
                    for i in range(features_to_add):
                        feature_name = f"missing_feature_{i}"
                        X_realtime[feature_name] = 0
                    
                    print(f"Added features. New shape: {X_realtime.shape}")
                    
                elif current_num_features > expected_num_features:
                    # Supprimer des features en trop
                    features_to_remove = current_num_features - expected_num_features
                    print(f"Removing {features_to_remove} extra features")
                    # Supprimer les dernières colonnes
                    X_realtime = X_realtime.iloc[:, :-features_to_remove]
                    print(f"Removed features. New shape: {X_realtime.shape}")
            else:
                print("Cannot determine model's expected number of features")
        
        print(f"Final feature shape for prediction: {X_realtime.shape}")
        
        # Faire la prédiction
        prediction_encoded = model.predict(X_realtime)[0]
        prediction_proba = model.predict_proba(X_realtime)[0]
        
        # Décoder la prédiction
        predicted_weather = label_encoder.inverse_transform([prediction_encoded])[0]
        max_proba = max(prediction_proba)
        
        print(f"🔮 PREDICTION RESULTS:")
        print(f"   Predicted weather: {predicted_weather}")
        print(f"   Confidence: {max_proba:.2%}")
        print(f"   Actual weather: {preprocessed_info['actual_weather']}")
        
        # Calculer toutes les probabilités par classe
        class_probabilities = {}
        
        print(f"Label encoder classes: {label_encoder.classes_}")
        print(f"Number of classes in label encoder: {len(label_encoder.classes_)}")
        print(f"Prediction probabilities shape: {prediction_proba.shape}")
        print(f"Prediction probabilities: {prediction_proba}")
        
        # S'assurer que nous n'essayons pas d'accéder à un index hors limites
        max_classes = min(len(label_encoder.classes_), len(prediction_proba))
        
        for i in range(max_classes):
            class_name = label_encoder.classes_[i]
            prob_value = float(prediction_proba[i])
            class_probabilities[class_name] = prob_value
            
        # Si il y a plus de classes dans le label encoder que de probabilités, 
        # mettre les probabilités manquantes à 0
        if len(label_encoder.classes_) > len(prediction_proba):
            for i in range(len(prediction_proba), len(label_encoder.classes_)):
                class_name = label_encoder.classes_[i]
                class_probabilities[class_name] = 0.0
                print(f"Set missing probability for class '{class_name}' to 0.0")
        
        print(f"   All probabilities: {class_probabilities}")
        
        # Vérifier si la prédiction est correcte
        is_correct = predicted_weather == preprocessed_info['actual_weather']
        print(f"   Prediction accuracy: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        
        # Préparer les résultats
        prediction_results = {
            'timestamp': preprocessed_info['timestamp'],
            'actual_weather': preprocessed_info['actual_weather'],
            'predicted_weather': predicted_weather,
            'confidence': float(max_proba),
            'is_correct': is_correct,
            'class_probabilities': class_probabilities,
            'model_run_id': model_info['run_id'],
            'original_features': preprocessed_info['original_features']
        }
        
        # Sauvegarder les résultats
        results_path = f"{PREDICTIONS_PATH}/prediction_results_{model_info['timestamp']}.json"
        with open(results_path, 'w') as f:
            json.dump(prediction_results, f, indent=2)
        
        print(f"✓ Prediction results saved to: {results_path}")
        
        # S'assurer que toutes les valeurs de retour sont sérialisables pour XCom
        return safe_json_serialize(prediction_results)
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        raise


def log_prediction_to_mlflow(**context):
    """Tâche 5: Logger la prédiction dans MLflow pour le monitoring"""
    print("Logging prediction results to MLflow...")
    
    # Setup MLflow
    setup_mlflow()
    
    # Récupérer les résultats de prédiction
    ti = context['ti']
    prediction_results = ti.xcom_pull(task_ids='make_prediction')
    
    # Créer un nouveau run MLflow pour les prédictions
    experiment = mlflow.get_experiment_by_name("Meteo")
    run_name = f"RealTime_Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
        # Logger les métriques de prédiction
        mlflow.log_metric("prediction_confidence", prediction_results['confidence'])
        mlflow.log_metric("prediction_accuracy", 1.0 if prediction_results['is_correct'] else 0.0)
        
        # Logger les paramètres
        mlflow.log_param("prediction_type", "real_time")
        mlflow.log_param("data_source", "openweather_api")
        mlflow.log_param("model_run_id", prediction_results['model_run_id'])
        mlflow.log_param("actual_weather", prediction_results['actual_weather'])
        mlflow.log_param("predicted_weather", prediction_results['predicted_weather'])
        
        # Logger les probabilités pour chaque classe
        for class_name, prob in prediction_results['class_probabilities'].items():
            mlflow.log_metric(f"prob_{class_name}", prob)
        
        # Logger les features originales
        original_features = prediction_results['original_features']
        for key, value in original_features.items():
            if isinstance(value, (int, float)) and key != 'datetime':
                mlflow.log_metric(f"feature_{key}", value)
        
        # Logger les résultats en tant qu'artefact
        results_text = f"""Real-Time Weather Prediction Results
Timestamp: {prediction_results['timestamp']}
Actual Weather: {prediction_results['actual_weather']}
Predicted Weather: {prediction_results['predicted_weather']}
Confidence: {prediction_results['confidence']:.2%}
Accuracy: {'CORRECT' if prediction_results['is_correct'] else 'INCORRECT'}

Class Probabilities:
{json.dumps(prediction_results['class_probabilities'], indent=2)}

Original Weather Features:
{json.dumps(original_features, indent=2, default=str)}
        """
        
        mlflow.log_text(results_text, "prediction_results.txt")
        
        print(f"✓ Prediction logged to MLflow run: {run.info.run_id}")
        
        return run.info.run_id


# Définition des tâches
task_fetch_weather = PythonOperator(
    task_id='fetch_weather_data',
    python_callable=fetch_weather_data,
    dag=dag,
)

task_preprocess_realtime = PythonOperator(
    task_id='preprocess_realtime_data',
    python_callable=preprocess_realtime_data,
    dag=dag,
)

task_load_model = PythonOperator(
    task_id='load_model_from_mlflow',
    python_callable=load_model_from_mlflow,
    dag=dag,
)

task_predict = PythonOperator(
    task_id='make_prediction',
    python_callable=make_prediction,
    dag=dag,
)

task_log_prediction = PythonOperator(
    task_id='log_prediction_to_mlflow',
    python_callable=log_prediction_to_mlflow,
    dag=dag,
)

# Définition des dépendances
task_fetch_weather >> task_preprocess_realtime
task_load_model >> task_predict
task_preprocess_realtime >> task_predict >> task_log_prediction

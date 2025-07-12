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
import boto3
from botocore.exceptions import ClientError

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
    schedule_interval=timedelta(minutes=5),  # Exécution toutes les 5 minutes
    catchup=False,
    tags=['ml', 'weather', 'prediction', 'real-time', 'openweather', 'mlflow'],
)

# Configuration des chemins et variables
MODEL_PATH = '/opt/airflow/models'
PREDICTIONS_PATH = '/opt/airflow/predictions'
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/3.0/onecall"
API_KEY = "8021a55eaa75f382697bb1956b2589b4"

# Configuration des villes pour les prédictions
CITIES = {
    'paris': {'lat': 48.8566, 'lon': 2.3522, 'name': 'Paris'},
    'toulouse': {'lat': 43.6047, 'lon': 1.4442, 'name': 'Toulouse'},
    'lyon': {'lat': 45.7640, 'lon': 4.8357, 'name': 'Lyon'},
    'marseille': {'lat': 43.2965, 'lon': 5.3698, 'name': 'Marseille'},
    'nantes': {'lat': 47.2184, 'lon': -1.5536, 'name': 'Nantes'}
}

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


def fetch_weather_data(**context):
    """Tâche 1: Récupérer les données météo en temps réel depuis OpenWeather API pour toutes les villes"""
    print("Fetching real-time weather data from OpenWeather API for all cities...")
    
    all_cities_data = {}
    
    for city_key, city_info in CITIES.items():
        print(f"\nFetching weather data for {city_info['name']}...")
        
        # Construire l'URL de l'API pour cette ville
        params = {
            'lat': city_info['lat'],
            'lon': city_info['lon'],
            'appid': API_KEY,
            'units': 'metric',  # Températures en Celsius
            'exclude': 'minutely,alerts'  # Exclure les données non nécessaires
        }
        
        try:
            # Faire l'appel API
            response = requests.get(OPENWEATHER_API_URL, params=params, timeout=30)
            response.raise_for_status()
            
            weather_data = response.json()
            print(f"API Response received successfully for {city_info['name']}")
            
            # Extraire les données actuelles
            current = weather_data['current']
            current_time = datetime.fromtimestamp(current['dt'])
            
            # Préparer les données pour la prédiction
            weather_features = {
                'city': city_info['name'],
                'city_key': city_key,
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
                'sunset': weather_data['current'].get('sunset', 0),
                'lat': city_info['lat'],
                'lon': city_info['lon']
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
            
            print(f"Current weather in {city_info['name']}: {weather_features['weather_main']} - {weather_features['weather_description']}")
            print(f"Temperature: {weather_features['temp']}°C")
            print(f"Humidity: {weather_features['humidity']}%")
            
            # Sauvegarder les données brutes
            os.makedirs(PREDICTIONS_PATH, exist_ok=True)
            raw_data_path = f"{PREDICTIONS_PATH}/raw_weather_data_{city_key}_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
            
            # Préparer les données pour la sérialisation JSON (convertir datetime en string)
            weather_features_serializable = weather_features.copy()
            weather_features_serializable['datetime'] = current_time.isoformat()
            
            # Sérialiser de manière sécurisée toute la réponse API
            safe_weather_data = safe_json_serialize(weather_data)
            
            with open(raw_data_path, 'w') as f:
                json.dump({
                    'timestamp': current_time.isoformat(),
                    'city': city_info['name'],
                    'city_key': city_key,
                    'weather_features': weather_features_serializable,
                    'full_response': safe_weather_data
                }, f, indent=2)
            
            # Stocker les données de cette ville
            all_cities_data[city_key] = {
                'raw_data_path': raw_data_path,
                'weather_features': weather_features_serializable,
                'city_info': city_info
            }
            
            print(f"✓ Weather data saved for {city_info['name']}")
            
        except Exception as e:
            print(f"❌ Error fetching weather data for {city_info['name']}: {str(e)}")
            # Continuer avec les autres villes même si une échoue
            continue
    
    if not all_cities_data:
        raise ValueError("Could not fetch weather data for any city")
    
    print(f"\n✅ Successfully fetched weather data for {len(all_cities_data)} cities: {list(all_cities_data.keys())}")
    return all_cities_data
    
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
    """Tâche 2: Préprocesser les données en temps réel pour la prédiction de toutes les villes"""
    print("Preprocessing real-time weather data for all cities...")
    
    # Récupérer les données de la tâche précédente
    ti = context['ti']
    all_cities_data = ti.xcom_pull(task_ids='fetch_weather_data')
    
    if not all_cities_data:
        raise ValueError("No weather data received from fetch_weather_data task")
    
    processed_cities = {}
    
    for city_key, city_data in all_cities_data.items():
        print(f"\nProcessing data for {city_data['city_info']['name']}...")
        weather_features = city_data['weather_features']
        
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
        
        print(f"  Original weather: {current_weather}, Mapped to: {mapped_weather}")
        
        # Supprimer les colonnes non nécessaires pour la prédiction
        columns_to_drop = [
            'datetime', 'weather_main', 'weather_description', 'weather_main_mapped',
            'sunrise', 'sunset', 'snow_1h', 'city', 'city_key'
        ]
        
        available_columns = [col for col in columns_to_drop if col in df.columns]
        feature_columns = [col for col in df.columns if col not in available_columns]
        
        X_realtime = df[feature_columns]
        
        print(f"  Preprocessed features shape: {X_realtime.shape}")
        print(f"  Features: {list(X_realtime.columns)}")
        
        # Vérifier que nous avons des valeurs numériques valides
        for col in X_realtime.columns:
            if X_realtime[col].iloc[0] is None or pd.isna(X_realtime[col].iloc[0]):
                print(f"  Warning: Feature {col} has null value, setting to 0")
                X_realtime[col] = X_realtime[col].fillna(0)
        
        # Sauvegarder les données préprocessées pour cette ville
        timestamp_str = weather_features['datetime'].replace(':', '-')
        preprocessed_path = f"{PREDICTIONS_PATH}/preprocessed_data_{city_key}_{timestamp_str}.pkl"
        X_realtime.to_pickle(preprocessed_path)
        
        processed_cities[city_key] = {
            'preprocessed_path': preprocessed_path,
            'actual_weather': mapped_weather,
            'timestamp': weather_features['datetime'],
            'original_features': weather_features,
            'city_info': city_data['city_info']
        }
        
        print(f"  ✓ Data processed and saved for {city_data['city_info']['name']}")
    
    print(f"\n✅ Successfully processed data for {len(processed_cities)} cities")
    return processed_cities


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
    """Tâche 4: Faire la prédiction avec le modèle chargé pour toutes les villes"""
    print("Making weather predictions for all cities...")
    
    # Récupérer les données des tâches précédentes
    ti = context['ti']
    all_cities_processed = ti.xcom_pull(task_ids='preprocess_realtime_data')
    model_info = ti.xcom_pull(task_ids='load_model_from_mlflow')
    
    if not all_cities_processed:
        raise ValueError("No processed data received from preprocess_realtime_data task")
    
    # Charger le modèle et le label encoder
    with open(model_info['model_path'], 'rb') as f:
        model = pickle.load(f)
    
    with open(model_info['label_encoder_path'], 'rb') as f:
        label_encoder = pickle.load(f)
    
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
    
    all_predictions = {}
    
    for city_key, city_data in all_cities_processed.items():
        print(f"\n🔮 Making prediction for {city_data['city_info']['name']}...")
        
        # Charger les données préprocessées pour cette ville
        X_realtime = pd.read_pickle(city_data['preprocessed_path'])
        
        print(f"Real-time data features for {city_data['city_info']['name']}: {list(X_realtime.columns)}")
        print(f"Real-time data shape: {X_realtime.shape}")
        
        try:
            # Vérifier la compatibilité des features
            current_features = X_realtime.columns.tolist()
            
            if expected_features is not None:
                # Utiliser les features d'entraînement comme référence
                missing_features = set(expected_features) - set(current_features)
                extra_features = set(current_features) - set(expected_features)
                
                if missing_features:
                    print(f"  Adding missing features with default values: {missing_features}")
                    for feature in missing_features:
                        X_realtime[feature] = 0
                
                if extra_features:
                    print(f"  Removing extra features: {extra_features}")
                
                # Garder seulement les features d'entraînement dans le bon ordre
                X_realtime = X_realtime[expected_features]
                print(f"  Features adjusted to match training data")
                
            # Si on n'a pas pu déterminer les features attendues, essayer de deviner par le nombre de colonnes
            if expected_features is None:
                if hasattr(model, 'feature_names_in_'):
                    expected_features = list(model.feature_names_in_)
                elif hasattr(model, 'n_features_in_'):
                    # Prendre les n premières colonnes
                    n = model.n_features_in_
                    if X_realtime.shape[1] > n:
                        print(f"  [AUTO] Trimming features to first {n} columns for model input")
                        X_realtime = X_realtime.iloc[:, :n]
                    elif X_realtime.shape[1] < n:
                        raise ValueError(f"Not enough features for model: expected {n}, got {X_realtime.shape[1]}")
                else:
                    raise ValueError("Cannot determine expected features for the model. Please check your training pipeline.")

            # Si on a la liste des features attendues, forcer la sélection
            if expected_features is not None:
                missing_features = set(expected_features) - set(X_realtime.columns)
                extra_features = set(X_realtime.columns) - set(expected_features)
                if missing_features:
                    print(f"  [AUTO] Adding missing features with default values: {missing_features}")
                    for feature in missing_features:
                        X_realtime[feature] = 0
                if extra_features:
                    print(f"  [AUTO] Removing extra features: {extra_features}")
                X_realtime = X_realtime[expected_features]
                print(f"  [AUTO] Final feature shape for prediction: {X_realtime.shape}")
            
            print(f"  Final feature shape for prediction: {X_realtime.shape}")
            
            # Faire la prédiction
            prediction_encoded = model.predict(X_realtime)[0]
            prediction_proba = model.predict_proba(X_realtime)[0]
            
            # Décoder la prédiction
            predicted_weather = label_encoder.inverse_transform([prediction_encoded])[0]
            max_proba = max(prediction_proba)
            
            print(f"  🔮 PREDICTION RESULTS for {city_data['city_info']['name']}:")
            print(f"     Predicted weather: {predicted_weather}")
            print(f"     Confidence: {max_proba:.2%}")
            print(f"     Actual weather: {city_data['actual_weather']}")
            
            # Calculer toutes les probabilités par classe
            class_probabilities = {}
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
            
            # Vérifier si la prédiction est correcte
            is_correct = predicted_weather == city_data['actual_weather']
            print(f"     Prediction accuracy: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            
            # Préparer les résultats pour cette ville
            city_prediction_results = {
                'city': city_data['city_info']['name'],
                'city_key': city_key,
                'timestamp': city_data['timestamp'],
                'actual_weather': city_data['actual_weather'],
                'predicted_weather': predicted_weather,
                'confidence': float(max_proba),
                'is_correct': is_correct,
                'class_probabilities': class_probabilities,
                'model_run_id': model_info['run_id'],
                'original_features': city_data['original_features'],
                'location': {
                    'latitude': city_data['city_info']['lat'],
                    'longitude': city_data['city_info']['lon']
                }
            }
            
            # Sauvegarder les résultats pour cette ville
            timestamp_str = city_data['timestamp'].replace(':', '-')
            results_path = f"{PREDICTIONS_PATH}/prediction_results_{city_key}_{timestamp_str}.json"
            with open(results_path, 'w') as f:
                json.dump(city_prediction_results, f, indent=2)
            
            print(f"  ✓ Prediction results saved for {city_data['city_info']['name']}")
            
            all_predictions[city_key] = city_prediction_results
            
        except Exception as e:
            print(f"  ❌ Error making prediction for {city_data['city_info']['name']}: {str(e)}")
            # Continuer avec les autres villes même si une échoue
            continue
    
    if not all_predictions:
        raise ValueError("Could not make predictions for any city")
    
    print(f"\n✅ Successfully made predictions for {len(all_predictions)} cities")
    
    # S'assurer que toutes les valeurs de retour sont sérialisables pour XCom
    return safe_json_serialize(all_predictions)


def save_prediction_to_s3(**context):
    """Tâche 6: Sauvegarder les prédictions dans un fichier CSV sur S3 (append mode)"""
    print("Saving prediction results to CSV on S3...")
    
    # Récupérer les résultats de prédiction de la tâche précédente
    ti = context['ti']
    all_predictions = ti.xcom_pull(task_ids='make_prediction')
    
    if not all_predictions:
        raise ValueError("No prediction results found to save to S3")
    
    try:
        # Charger les credentials AWS depuis la connexion Airflow
        aws_conn = BaseHook.get_connection('aws_default')
        
        # Configurer le client S3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_conn.login,
            aws_secret_access_key=aws_conn.password,
            region_name=aws_conn.extra_dejson.get('region_name', 'eu-west-2')
        )
        
        bucket_name = 'projetmeteo'
        csv_file_name = 'meteo_predict/weather_predictions.csv'
        
        # Préparer les nouvelles données pour le CSV
        new_rows = []
        current_timestamp = datetime.now().isoformat()
        
        for city_key, prediction_results in all_predictions.items():
            # Convertir timestamp en format ISO si nécessaire
            timestamp = prediction_results['timestamp']
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_timestamp = dt.isoformat()
                except:
                    formatted_timestamp = timestamp
            else:
                formatted_timestamp = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
            
            new_row = {
                'timestamp': formatted_timestamp,
                'latitude': prediction_results['location']['latitude'],
                'longitude': prediction_results['location']['longitude'],
                'ville': prediction_results['city'],
                'prediction': prediction_results['predicted_weather'],
                'valeur_reelle': prediction_results['actual_weather'],
                'confidence': prediction_results['confidence'],
                'is_correct': prediction_results['is_correct'],
                'dag_run_id': context['run_id'],
                'execution_date': context['execution_date'].isoformat()
            }
            new_rows.append(new_row)
            
            print(f"  📊 {prediction_results['city']}: {prediction_results['predicted_weather']} vs {prediction_results['actual_weather']} (confidence: {prediction_results['confidence']:.2%})")
        
        # Créer un DataFrame avec les nouvelles données
        new_df = pd.DataFrame(new_rows)
        
        # Essayer de télécharger le fichier CSV existant depuis S3
        existing_df = None
        file_exists = False
        
        try:
            print(f"Checking if CSV file exists: {csv_file_name}")
            response = s3_client.get_object(Bucket=bucket_name, Key=csv_file_name)
            existing_csv_content = response['Body'].read().decode('utf-8')
            
            # Lire le CSV existant
            from io import StringIO
            existing_df = pd.read_csv(StringIO(existing_csv_content))
            file_exists = True
            print(f"✓ Existing CSV found with {len(existing_df)} rows")
            
        except s3_client.exceptions.NoSuchKey:
            print("📄 CSV file doesn't exist yet, will create new one")
            file_exists = False
        except Exception as e:
            print(f"⚠️  Error reading existing CSV: {str(e)}, will create new file")
            file_exists = False
        
        # Combiner les données existantes avec les nouvelles
        if file_exists and existing_df is not None and not existing_df.empty:
            # Append les nouvelles données
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"📈 Appending {len(new_df)} new rows to existing {len(existing_df)} rows")
        else:
            # Créer un nouveau fichier
            combined_df = new_df
            print(f"🆕 Creating new CSV with {len(new_df)} rows")
        
        # Trier par timestamp pour garder l'ordre chronologique
        combined_df = combined_df.sort_values('timestamp')
        
        # Sauvegarder en tant que CSV
        csv_buffer = combined_df.to_csv(index=False)
        
        # Uploader le CSV mis à jour vers S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=csv_file_name,
            Body=csv_buffer.encode('utf-8'),
            ContentType='text/csv',
            Metadata={
                'dag-id': 'real_time_weather_prediction',
                'total-rows': str(len(combined_df)),
                'cities': ','.join([pred['city'] for pred in all_predictions.values()]),
                'last-updated': current_timestamp,
                'format': 'csv'
            }
        )
        
        s3_url = f"s3://{bucket_name}/{csv_file_name}"
        
        # Statistiques pour le rapport
        total_predictions = len(all_predictions)
        correct_predictions = sum(1 for pred in all_predictions.values() if pred['is_correct'])
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"\n✅ CSV predictions successfully saved to S3:")
        print(f"   📍 S3 URL: {s3_url}")
        print(f"   📊 Total rows in CSV: {len(combined_df)}")
        print(f"   🆕 New rows added: {len(new_df)}")
        print(f"   🎯 Current batch accuracy: {accuracy:.2%}")
        print(f"   🏙️  Cities: {', '.join([pred['city'] for pred in all_predictions.values()])}")
        
        # Afficher un échantillon des dernières lignes
        print(f"\n📋 Sample of latest predictions:")
        latest_rows = combined_df.tail(min(5, len(new_df)))
        for _, row in latest_rows.iterrows():
            print(f"   {row['ville']}: {row['prediction']} vs {row['valeur_reelle']} ({'✓' if row['is_correct'] else '✗'})")
        
        return {
            'csv_s3_url': s3_url,
            'csv_file_name': csv_file_name,
            'total_rows': len(combined_df),
            'new_rows_added': len(new_df),
            'current_batch_accuracy': accuracy,
            'cities_processed': list(all_predictions.keys()),
            'file_existed': file_exists
        }
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            print(f"❌ Error: Bucket '{bucket_name}' does not exist")
        elif error_code == 'AccessDenied':
            print(f"❌ Error: Access denied to bucket '{bucket_name}'. Check your AWS credentials.")
        else:
            print(f"❌ AWS S3 Error: {error_code} - {e.response['Error']['Message']}")
        raise
        
    except Exception as e:
        print(f"❌ Error saving predictions to S3: {str(e)}")
        raise


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

task_save_prediction_s3 = PythonOperator(
    task_id='save_prediction_to_s3',
    python_callable=save_prediction_to_s3,
    dag=dag,
)

# Définition des dépendances
task_fetch_weather >> task_preprocess_realtime
task_load_model >> task_predict
task_preprocess_realtime >> task_predict >> task_save_prediction_s3

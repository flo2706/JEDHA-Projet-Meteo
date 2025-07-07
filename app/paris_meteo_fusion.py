# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import mlflow, os
from xgboost import XGBClassifier
import mlflow.xgboost
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, 
                           accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report)
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Load AWS credentials from credentials.txt file
def load_aws_credentials():
    try:
        with open('credentials.txt', 'r') as f:
            for line in f:
                if line.strip() and '=' in line:
                    key, value = line.strip().split(' = ', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        print("credentials.txt file not found")

load_aws_credentials()

os.environ["APP_URI"] = "https://6a44-91-164-131-62.ngrok-free.app" #MLFLOW

# Load Fichier CSV 

paris_df = pd.read_csv("weather_paris.csv")

#fusionner les classes
paris_df['weather_main'] = paris_df['weather_main'].replace({'Drizzle': 'Rain', 'Mist': 'Fog'})

#Split le temps en detail

paris_df['datetime'] = pd.to_datetime(paris_df['datetime'])
paris_df['hour'] = paris_df['datetime'].dt.hour
paris_df['day'] = paris_df['datetime'].dt.day
paris_df['month'] = paris_df['datetime'].dt.month
paris_df['weekday'] = paris_df['datetime'].dt.weekday
paris_df['is_weekend'] = paris_df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
paris_df['hour_sin'] = np.sin(2 * np.pi * paris_df['hour'] / 24)
paris_df['hour_cos'] = np.cos(2 * np.pi * paris_df['hour'] / 24)
paris_df['month_sin'] = np.sin(2 * np.pi * paris_df['month'] / 12)
paris_df['month_cos'] = np.cos(2 * np.pi * paris_df['month'] / 12)

paris_df = paris_df.drop(['datetime'], axis=1)
paris_df = paris_df.drop(['weather_description'], axis=1)

# --- Encodage de la variable cible weather_main ---
le = LabelEncoder()
paris_df['weather_main_encoded'] = le.fit_transform(paris_df['weather_main'])

# --- Separer X et y ---
X = paris_df.drop(columns=['weather_main', 'weather_main_encoded'])  # features
Y = paris_df['weather_main_encoded']  # target encodée

# --- Train/test split ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# --- MLflow setup ---
EXPERIMENT_NAME = "Meteo"
mlflow.set_tracking_uri(os.environ["APP_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
run_name = "Run_XGBClassifier_" + datetime.now().strftime("%Y%m%d_%H%M%S")


mlflow.sklearn.autolog()  # active autolog pour sklearn (attention : ici tu utilises xgboost, mais autolog scikit-learn peut fonctionner aussi)

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
    # Initialiser et entraîner le modèle XGBoost Classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, Y_train)

    # Faire des prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculer les probabilités pour AUC-ROC
    y_train_pred_proba = model.predict_proba(X_train)
    y_test_pred_proba = model.predict_proba(X_test)

    # Calculer les métriques de classification pour l'ensemble d'entraînement
    accuracy_train = accuracy_score(Y_train, y_train_pred)
    precision_train = precision_score(Y_train, y_train_pred, average='weighted')
    recall_train = recall_score(Y_train, y_train_pred, average='weighted')
    f1_train = f1_score(Y_train, y_train_pred, average='weighted')

    # Calculer les métriques de classification pour l'ensemble de test
    accuracy_test = accuracy_score(Y_test, y_test_pred)
    precision_test = precision_score(Y_test, y_test_pred, average='weighted')
    recall_test = recall_score(Y_test, y_test_pred, average='weighted')
    f1_test = f1_score(Y_test, y_test_pred, average='weighted')

    # Enregistrer les métriques dans MLflow
    mlflow.log_metric("Accuracy Train", accuracy_train)
    mlflow.log_metric("Precision Train", precision_train)
    mlflow.log_metric("Recall Train", recall_train)
    mlflow.log_metric("F1 Score Train", f1_train)


    mlflow.log_metric("Accuracy Test", accuracy_test)
    mlflow.log_metric("Precision Test", precision_test)
    mlflow.log_metric("Recall Test", recall_test)
    mlflow.log_metric("F1 Score Test", f1_test)
    

    # Enregistrer la matrice de confusion et le rapport de classification pour l'ensemble de test
    confusion_matrix_test = confusion_matrix(Y_test, y_test_pred)
    classification_report_test = classification_report(Y_test, y_test_pred)

    mlflow.log_text(str(confusion_matrix_test), "confusion_matrix_test.txt")
    mlflow.log_text(classification_report_test, "classification_report_test.txt")

    # Enregistrer le modèle dans MLflow
    mlflow.xgboost.log_model(model, "model")


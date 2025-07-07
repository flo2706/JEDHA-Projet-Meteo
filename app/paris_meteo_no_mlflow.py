# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, 
                           accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report)
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


def load_and_preprocess_data(file_path="data/weather_paris.csv"):
    """Load and preprocess the weather data."""
    # Load Fichier CSV 
    paris_df = pd.read_csv(file_path)
    
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
    
    return paris_df


def encode_target_variable(df):
    """Encode the target variable weather_main."""
    le = LabelEncoder()
    df['weather_main_encoded'] = le.fit_transform(df['weather_main'])
    return df, le


def prepare_features_and_target(df):
    """Separate features and target variable."""
    X = df.drop(columns=['weather_main', 'weather_main_encoded'])  # features
    Y = df['weather_main_encoded']  # target encodée
    return X, Y


def split_data(X, Y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)


def train_model(X_train, Y_train):
    """Train XGBoost classifier."""
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, Y_train)
    return model


def make_predictions(model, X_train, X_test):
    """Make predictions on train and test sets."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_pred_proba = model.predict_proba(X_train)
    y_test_pred_proba = model.predict_proba(X_test)
    return y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba


def calculate_metrics(Y_true, y_pred):
    """Calculate classification metrics."""
    metrics = {
        'accuracy': accuracy_score(Y_true, y_pred),
        'precision': precision_score(Y_true, y_pred, average='weighted'),
        'recall': recall_score(Y_true, y_pred, average='weighted'),
        'f1': f1_score(Y_true, y_pred, average='weighted')
    }
    return metrics


def print_results(train_metrics, test_metrics, Y_test, y_test_pred):
    """Print all results."""
    print("=== RÉSULTATS D'ENTRAÎNEMENT ===")
    print(f"Accuracy Train: {train_metrics['accuracy']:.4f}")
    print(f"Precision Train: {train_metrics['precision']:.4f}")
    print(f"Recall Train: {train_metrics['recall']:.4f}")
    print(f"F1 Score Train: {train_metrics['f1']:.4f}")

    print("\n=== RÉSULTATS DE TEST ===")
    print(f"Accuracy Test: {test_metrics['accuracy']:.4f}")
    print(f"Precision Test: {test_metrics['precision']:.4f}")
    print(f"Recall Test: {test_metrics['recall']:.4f}")
    print(f"F1 Score Test: {test_metrics['f1']:.4f}")

    # Afficher la matrice de confusion et le rapport de classification pour l'ensemble de test
    confusion_matrix_test = confusion_matrix(Y_test, y_test_pred)
    classification_report_test = classification_report(Y_test, y_test_pred)

    print("\n=== MATRICE DE CONFUSION (TEST) ===")
    print(confusion_matrix_test)

    print("\n=== RAPPORT DE CLASSIFICATION (TEST) ===")
    print(classification_report_test)


def main():
    """Main function to run the complete pipeline."""
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Encode target variable
    df, label_encoder = encode_target_variable(df)
    
    # Prepare features and target
    X, Y = prepare_features_and_target(df)
    
    # Split data
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    
    # Train model
    model = train_model(X_train, Y_train)
    
    # Make predictions
    y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba = make_predictions(model, X_train, X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(Y_train, y_train_pred)
    test_metrics = calculate_metrics(Y_test, y_test_pred)
    
    # Print results
    print_results(train_metrics, test_metrics, Y_test, y_test_pred)
    
    return model, label_encoder, train_metrics, test_metrics


if __name__ == "__main__":
    main()

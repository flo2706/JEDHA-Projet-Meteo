# test_paris_meteo_no_mlflow.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Ajouter le répertoire parent au path pour pouvoir importer le module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paris_meteo_no_mlflow import (
    load_and_preprocess_data,
    encode_target_variable,
    prepare_features_and_target,
    split_data,
    train_model,
    make_predictions,
    calculate_metrics
)


@pytest.fixture
def sample_weather_data():
    """Fixture pour créer des données météo d'exemple."""
    data = {
        'datetime': ['2023-01-01 12:00:00', '2023-01-02 15:30:00', '2023-01-03 08:15:00', '2023-01-04 20:45:00'],
        'weather_main': ['Clear', 'Rain', 'Drizzle', 'Mist'],
        'weather_description': ['clear sky', 'light rain', 'drizzle', 'mist'],
        'temp': [15.5, 12.3, 8.7, 10.2],
        'humidity': [65, 80, 90, 85],
        'pressure': [1013, 1008, 1005, 1010],
        'wind_speed': [3.2, 5.1, 2.8, 4.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_weather_data):
    """Fixture pour créer un fichier CSV temporaire."""
    csv_file = tmp_path / "test_weather.csv"
    sample_weather_data.to_csv(csv_file, index=False)
    return str(csv_file)


class TestDataPreprocessing:
    """Tests pour les fonctions de préprocessing des données."""

    def test_load_and_preprocess_data(self, sample_csv_file):
        """Test de la fonction load_and_preprocess_data."""
        df = load_and_preprocess_data(sample_csv_file)
        
        # Vérifier que le dataframe n'est pas vide
        assert not df.empty
        
        # Vérifier que les colonnes datetime et weather_description ont été supprimées
        assert 'datetime' not in df.columns
        assert 'weather_description' not in df.columns
        
        # Vérifier que les nouvelles colonnes temporelles ont été créées
        expected_columns = ['hour', 'day', 'month', 'weekday', 'is_weekend', 
                           'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        for col in expected_columns:
            assert col in df.columns
        
        # Vérifier la fusion des classes météo
        assert 'Rain' in df['weather_main'].values
        assert 'Fog' in df['weather_main'].values
        assert 'Drizzle' not in df['weather_main'].values
        assert 'Mist' not in df['weather_main'].values

    def test_weather_class_fusion(self, sample_csv_file):
        """Test de la fusion des classes météo."""
        df = load_and_preprocess_data(sample_csv_file)
        
        # Vérifier que Drizzle a été transformé en Rain
        # et que Mist a été transformé en Fog
        unique_weather = df['weather_main'].unique()
        assert 'Drizzle' not in unique_weather
        assert 'Mist' not in unique_weather

    def test_temporal_features(self, sample_csv_file):
        """Test de la création des features temporelles."""
        df = load_and_preprocess_data(sample_csv_file)
        
        # Vérifier que les features sinusoïdales sont dans la bonne plage
        assert df['hour_sin'].min() >= -1
        assert df['hour_sin'].max() <= 1
        assert df['hour_cos'].min() >= -1
        assert df['hour_cos'].max() <= 1
        assert df['month_sin'].min() >= -1
        assert df['month_sin'].max() <= 1
        assert df['month_cos'].min() >= -1
        assert df['month_cos'].max() <= 1
        
        # Vérifier que is_weekend est binaire
        assert set(df['is_weekend'].unique()).issubset({0, 1})


class TestTargetEncoding:
    """Tests pour l'encodage de la variable cible."""

    def test_encode_target_variable(self, sample_csv_file):
        """Test de l'encodage de la variable cible."""
        df = load_and_preprocess_data(sample_csv_file)
        df_encoded, label_encoder = encode_target_variable(df)
        
        # Vérifier que la colonne encodée a été créée
        assert 'weather_main_encoded' in df_encoded.columns
        
        # Vérifier que l'encodage est cohérent
        assert len(df_encoded['weather_main_encoded'].unique()) <= len(df_encoded['weather_main'].unique())
        
        # Vérifier que le label encoder fonctionne
        assert hasattr(label_encoder, 'classes_')


class TestFeaturePreparation:
    """Tests pour la préparation des features."""

    def test_prepare_features_and_target(self, sample_csv_file):
        """Test de la séparation des features et de la cible."""
        df = load_and_preprocess_data(sample_csv_file)
        df_encoded, _ = encode_target_variable(df)
        X, Y = prepare_features_and_target(df_encoded)
        
        # Vérifier que les features ne contiennent pas les colonnes cibles
        assert 'weather_main' not in X.columns
        assert 'weather_main_encoded' not in X.columns
        
        # Vérifier les dimensions
        assert len(X) == len(Y)
        assert len(X) == len(df_encoded)


class TestDataSplitting:
    """Tests pour le splitting des données."""

    def test_split_data(self, sample_csv_file):
        """Test du splitting des données."""
        df = load_and_preprocess_data(sample_csv_file)
        df_encoded, _ = encode_target_variable(df)
        X, Y = prepare_features_and_target(df_encoded)
        
        X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.5, random_state=42)
        
        # Vérifier les dimensions
        assert len(X_train) + len(X_test) == len(X)
        assert len(Y_train) + len(Y_test) == len(Y)
        assert len(X_train) == len(Y_train)
        assert len(X_test) == len(Y_test)
        
        # Vérifier la proportion du test
        assert abs(len(X_test) / len(X) - 0.5) < 0.1  # Tolérance pour les petits datasets


class TestModelTraining:
    """Tests pour l'entraînement du modèle."""

    @patch('paris_meteo_no_mlflow.XGBClassifier')
    def test_train_model(self, mock_xgb, sample_csv_file):
        """Test de l'entraînement du modèle."""
        # Mock du modèle XGBoost
        mock_model = MagicMock()
        mock_xgb.return_value = mock_model
        
        # Préparer des données d'exemple
        X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        Y_train = pd.Series([0, 1, 0])
        
        # Entraîner le modèle
        model = train_model(X_train, Y_train)
        
        # Vérifier que XGBClassifier a été appelé avec les bons paramètres
        mock_xgb.assert_called_once_with(use_label_encoder=False, eval_metric='mlogloss')
        
        # Vérifier que fit a été appelé
        mock_model.fit.assert_called_once()


class TestPredictions:
    """Tests pour les prédictions."""

    def test_make_predictions(self):
        """Test de la fonction make_predictions."""
        # Mock du modèle
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        
        # Données d'exemple
        X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        X_test = pd.DataFrame({'feature1': [7, 8], 'feature2': [9, 10]})
        
        # Faire les prédictions
        y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba = make_predictions(
            mock_model, X_train, X_test
        )
        
        # Vérifier que les fonctions ont été appelées
        assert mock_model.predict.call_count == 2
        assert mock_model.predict_proba.call_count == 2
        
        # Vérifier le type des retours
        assert isinstance(y_train_pred, np.ndarray)
        assert isinstance(y_test_pred, np.ndarray)
        assert isinstance(y_train_pred_proba, np.ndarray)
        assert isinstance(y_test_pred_proba, np.ndarray)


class TestMetrics:
    """Tests pour le calcul des métriques."""

    def test_calculate_metrics(self):
        """Test du calcul des métriques."""
        Y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = calculate_metrics(Y_true, y_pred)
        
        # Vérifier que toutes les métriques sont présentes
        expected_keys = ['accuracy', 'precision', 'recall', 'f1']
        assert all(key in metrics for key in expected_keys)
        
        # Vérifier que les valeurs sont dans la bonne plage
        for key, value in metrics.items():
            assert 0 <= value <= 1
        
        # Vérifier l'accuracy calculée manuellement
        expected_accuracy = 4/5  # 4 bonnes prédictions sur 5
        assert abs(metrics['accuracy'] - expected_accuracy) < 0.001


class TestIntegration:
    """Tests d'intégration."""

    def test_file_not_found_error(self):
        """Test du comportement avec un fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            load_and_preprocess_data("fichier_inexistant.csv")

    def test_empty_dataframe_handling(self, tmp_path):
        """Test du comportement avec un DataFrame vide."""
        # Créer un fichier CSV vide
        empty_csv = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(empty_csv, index=False)
        
        with pytest.raises(Exception):  # Devrait lever une exception
            load_and_preprocess_data(str(empty_csv))


if __name__ == "__main__":
    pytest.main([__file__])

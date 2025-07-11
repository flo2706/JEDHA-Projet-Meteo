#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, NoCredentialsError

# Import du module à tester
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from data_to_s3 import S3DataUploader, main


class TestS3DataUploader:
    """Tests pour la classe S3DataUploader"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        # Variables d'environnement de test
        self.test_env = {
            'AWS_ACCESS_KEY_ID': 'test_access_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret_key',
            'AWS_DEFAULT_REGION': 'eu-west-2'
        }
        
        # Mock des variables d'environnement
        with patch.dict(os.environ, self.test_env):
            self.uploader = S3DataUploader()
    
    @patch.dict(os.environ, {
        'AWS_ACCESS_KEY_ID': 'test_key',
        'AWS_SECRET_ACCESS_KEY': 'test_secret',
        'AWS_DEFAULT_REGION': 'eu-west-2'
    })
    @patch('data_to_s3.boto3.client')
    def test_init_success(self, mock_boto3_client):
        """Test de l'initialisation réussie de S3DataUploader"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        uploader = S3DataUploader("test-bucket", "test-folder")
        
        assert uploader.bucket_name == "test-bucket"
        assert uploader.s3_folder == "test-folder"
        assert uploader.aws_access_key == "test_key"
        assert uploader.aws_secret_key == "test_secret"
        assert uploader.s3_client == mock_client
        
        mock_boto3_client.assert_called_once_with(
            's3',
            aws_access_key_id='test_key',
            aws_secret_access_key='test_secret',
            region_name='eu-west-2'
        )
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_missing_credentials(self):
        """Test de l'initialisation avec des credentials manquants"""
        with pytest.raises(ValueError, match="AWS credentials non trouvés"):
            S3DataUploader()
    
    @patch('data_to_s3.boto3.client')
    def test_create_s3_client_error(self, mock_boto3_client):
        """Test de gestion d'erreur lors de la création du client S3"""
        mock_boto3_client.side_effect = Exception("Erreur de connexion")
        
        with patch.dict(os.environ, self.test_env):
            with pytest.raises(Exception, match="Erreur de connexion"):
                S3DataUploader()
    
    @patch('data_to_s3.boto3.client')
    def test_check_bucket_exists_success(self, mock_boto3_client):
        """Test de vérification d'existence du bucket - succès"""
        mock_client = Mock()
        mock_client.head_bucket.return_value = {}
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
            result = uploader.check_bucket_exists()
        
        assert result is True
        mock_client.head_bucket.assert_called_once_with(Bucket='my-jedha-bucket')
    
    @patch('data_to_s3.boto3.client')
    def test_check_bucket_exists_not_found(self, mock_boto3_client):
        """Test de vérification d'existence du bucket - non trouvé"""
        mock_client = Mock()
        error_response = {'Error': {'Code': '404'}}
        mock_client.head_bucket.side_effect = ClientError(error_response, 'HeadBucket')
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
            result = uploader.check_bucket_exists()
        
        assert result is False
    
    @patch('data_to_s3.boto3.client')
    def test_check_bucket_exists_access_denied(self, mock_boto3_client):
        """Test de vérification d'existence du bucket - accès refusé"""
        mock_client = Mock()
        error_response = {'Error': {'Code': '403'}}
        mock_client.head_bucket.side_effect = ClientError(error_response, 'HeadBucket')
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
            result = uploader.check_bucket_exists()
        
        assert result is False
    
    @patch('data_to_s3.boto3.client')
    def test_upload_file_success(self, mock_boto3_client):
        """Test d'upload de fichier réussi"""
        mock_client = Mock()
        mock_client.upload_file.return_value = None
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
        
        # Création d'un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test data")
            temp_file_path = temp_file.name
        
        try:
            result = uploader.upload_file(temp_file_path)
            
            assert result is True
            mock_client.upload_file.assert_called_once_with(
                temp_file_path,
                'my-jedha-bucket',
                f'meteo_data/{Path(temp_file_path).name}'
            )
        finally:
            os.unlink(temp_file_path)
    
    @patch('data_to_s3.boto3.client')
    def test_upload_file_not_found(self, mock_boto3_client):
        """Test d'upload de fichier inexistant"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
        
        result = uploader.upload_file("fichier_inexistant.csv")
        
        assert result is False
        mock_client.upload_file.assert_not_called()
    
    @patch('data_to_s3.boto3.client')
    def test_upload_file_with_custom_s3_key(self, mock_boto3_client):
        """Test d'upload de fichier avec clé S3 personnalisée"""
        mock_client = Mock()
        mock_client.upload_file.return_value = None
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test data")
            temp_file_path = temp_file.name
        
        try:
            result = uploader.upload_file(temp_file_path, "custom/path/file.csv")
            
            assert result is True
            mock_client.upload_file.assert_called_once_with(
                temp_file_path,
                'my-jedha-bucket',
                'custom/path/file.csv'
            )
        finally:
            os.unlink(temp_file_path)
    
    @patch('data_to_s3.boto3.client')
    def test_upload_file_client_error(self, mock_boto3_client):
        """Test d'upload avec erreur client S3"""
        mock_client = Mock()
        error_response = {'Error': {'Code': '500', 'Message': 'Internal Error'}}
        mock_client.upload_file.side_effect = ClientError(error_response, 'UploadFile')
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test data")
            temp_file_path = temp_file.name
        
        try:
            result = uploader.upload_file(temp_file_path)
            assert result is False
        finally:
            os.unlink(temp_file_path)
    
    @patch('data_to_s3.boto3.client')
    def test_upload_file_no_credentials(self, mock_boto3_client):
        """Test d'upload avec erreur de credentials"""
        mock_client = Mock()
        mock_client.upload_file.side_effect = NoCredentialsError()
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test data")
            temp_file_path = temp_file.name
        
        try:
            result = uploader.upload_file(temp_file_path)
            assert result is False
        finally:
            os.unlink(temp_file_path)
    
    @patch('data_to_s3.boto3.client')
    def test_upload_directory_success(self, mock_boto3_client):
        """Test d'upload de répertoire réussi"""
        mock_client = Mock()
        mock_client.upload_file.return_value = None
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
        
        # Création d'un répertoire temporaire avec des fichiers
        with tempfile.TemporaryDirectory() as temp_dir:
            # Création de fichiers de test
            file1 = Path(temp_dir) / "file1.csv"
            file2 = Path(temp_dir) / "file2.txt"
            file1.write_text("data1")
            file2.write_text("data2")
            
            uploaded, total = uploader.upload_directory(temp_dir)
            
            assert uploaded == 2
            assert total == 2
            assert mock_client.upload_file.call_count == 2
    
    @patch('data_to_s3.boto3.client')
    def test_upload_directory_not_found(self, mock_boto3_client):
        """Test d'upload de répertoire inexistant"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
        
        uploaded, total = uploader.upload_directory("repertoire_inexistant")
        
        assert uploaded == 0
        assert total == 0
        mock_client.upload_file.assert_not_called()
    
    @patch('data_to_s3.boto3.client')
    def test_upload_directory_empty(self, mock_boto3_client):
        """Test d'upload de répertoire vide"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded, total = uploader.upload_directory(temp_dir)
            
            assert uploaded == 0
            assert total == 0
            mock_client.upload_file.assert_not_called()
    
    @patch('data_to_s3.boto3.client')
    def test_upload_directory_partial_failure(self, mock_boto3_client):
        """Test d'upload de répertoire avec échec partiel"""
        mock_client = Mock()
        # Premier appel réussit, deuxième échoue
        mock_client.upload_file.side_effect = [None, ClientError({}, 'UploadFile')]
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = Path(temp_dir) / "file1.csv"
            file2 = Path(temp_dir) / "file2.txt"
            file1.write_text("data1")
            file2.write_text("data2")
            
            uploaded, total = uploader.upload_directory(temp_dir)
            
            assert uploaded == 1
            assert total == 2
            assert mock_client.upload_file.call_count == 2
    
    @patch('data_to_s3.boto3.client')
    def test_list_s3_objects_success(self, mock_boto3_client):
        """Test de liste des objets S3 réussi"""
        mock_client = Mock()
        mock_response = {
            'Contents': [
                {'Key': 'meteo_data/file1.csv', 'Size': 1024},
                {'Key': 'meteo_data/file2.txt', 'Size': 512}
            ]
        }
        mock_client.list_objects_v2.return_value = mock_response
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
            uploader.list_s3_objects()  # Ne retourne rien, juste log
        
        mock_client.list_objects_v2.assert_called_once_with(
            Bucket='my-jedha-bucket',
            Prefix='meteo_data/'
        )
    
    @patch('data_to_s3.boto3.client')
    def test_list_s3_objects_empty(self, mock_boto3_client):
        """Test de liste des objets S3 vide"""
        mock_client = Mock()
        mock_response = {}  # Pas de 'Contents'
        mock_client.list_objects_v2.return_value = mock_response
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
            uploader.list_s3_objects()
        
        mock_client.list_objects_v2.assert_called_once_with(
            Bucket='my-jedha-bucket',
            Prefix='meteo_data/'
        )
    
    @patch('data_to_s3.boto3.client')
    def test_list_s3_objects_error(self, mock_boto3_client):
        """Test de liste des objets S3 avec erreur"""
        mock_client = Mock()
        error_response = {'Error': {'Code': '500'}}
        mock_client.list_objects_v2.side_effect = ClientError(error_response, 'ListObjects')
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, self.test_env):
            uploader = S3DataUploader()
            uploader.list_s3_objects()  # Ne doit pas lever d'exception
        
        mock_client.list_objects_v2.assert_called_once()


class TestMainFunction:
    """Tests pour la fonction main"""
    
    @patch('data_to_s3.S3DataUploader')
    @patch('data_to_s3.Path')
    def test_main_success(self, mock_path_class, mock_uploader_class):
        """Test de la fonction main avec succès"""
        # Mock du chemin avec __file__
        mock_file_path = Mock()
        mock_file_path.parent.parent = Path("/fake/project/root")
        mock_path_class.return_value = mock_file_path
        
        # Mock de l'uploader
        mock_uploader = Mock()
        mock_uploader.check_bucket_exists.return_value = True
        mock_uploader.upload_directory.return_value = (1, 1)  # Success
        mock_uploader_class.return_value = mock_uploader
        
        result = main()
        
        assert result is True
        mock_uploader.check_bucket_exists.assert_called_once()
        mock_uploader.upload_directory.assert_called_once()
        mock_uploader.list_s3_objects.assert_called_once()
    
    @patch('data_to_s3.S3DataUploader')
    @patch('data_to_s3.Path')
    def test_main_bucket_not_exists(self, mock_path_class, mock_uploader_class):
        """Test de la fonction main avec bucket inexistant"""
        mock_file_path = Mock()
        mock_file_path.parent.parent = Path("/fake/project/root")
        mock_path_class.return_value = mock_file_path
        
        mock_uploader = Mock()
        mock_uploader.check_bucket_exists.return_value = False
        mock_uploader_class.return_value = mock_uploader
        
        result = main()
        
        assert result is False
        mock_uploader.check_bucket_exists.assert_called_once()
        mock_uploader.upload_directory.assert_not_called()
    
    @patch('data_to_s3.S3DataUploader')
    @patch('data_to_s3.Path')
    def test_main_partial_upload(self, mock_path_class, mock_uploader_class):
        """Test de la fonction main avec upload partiel"""
        mock_file_path = Mock()
        mock_file_path.parent.parent = Path("/fake/project/root")
        mock_path_class.return_value = mock_file_path
        
        mock_uploader = Mock()
        mock_uploader.check_bucket_exists.return_value = True
        mock_uploader.upload_directory.return_value = (1, 2)  # Partial failure
        mock_uploader_class.return_value = mock_uploader
        
        result = main()
        
        assert result is False
        mock_uploader.check_bucket_exists.assert_called_once()
        mock_uploader.upload_directory.assert_called_once()
    
    @patch('data_to_s3.S3DataUploader')
    def test_main_exception(self, mock_uploader_class):
        """Test de la fonction main avec exception"""
        mock_uploader_class.side_effect = Exception("Erreur de test")
        
        result = main()
        
        assert result is False


class TestEnvironmentVariables:
    """Tests pour la gestion des variables d'environnement"""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('sys.exit')
    def test_missing_environment_variables(self, mock_exit):
        """Test avec variables d'environnement manquantes"""
        # Simulation de l'exécution du script principal
        # On vérifie juste que les variables manquantes sont détectées
        required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        # Tous les vars doivent être manquants
        assert len(missing_vars) == 2
        assert 'AWS_ACCESS_KEY_ID' in missing_vars
        assert 'AWS_SECRET_ACCESS_KEY' in missing_vars


@pytest.fixture
def sample_data_file():
    """Fixture pour créer un fichier de données de test"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("date,temperature,humidity\n")
        f.write("2023-01-01,15.5,65\n")
        f.write("2023-01-02,16.2,70\n")
        yield f.name
    os.unlink(f.name)


class TestIntegration:
    """Tests d'intégration"""
    
    @patch('data_to_s3.boto3.client')
    def test_end_to_end_upload(self, mock_boto3_client, sample_data_file):
        """Test d'intégration bout en bout"""
        mock_client = Mock()
        mock_client.head_bucket.return_value = {}
        mock_client.upload_file.return_value = None
        mock_client.list_objects_v2.return_value = {'Contents': []}
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret'
        }):
            uploader = S3DataUploader("test-bucket", "test-folder")
            
            # Test complet
            assert uploader.check_bucket_exists() is True
            assert uploader.upload_file(sample_data_file) is True
            uploader.list_s3_objects()  # Ne doit pas lever d'exception
            
            # Vérifications
            mock_client.head_bucket.assert_called_once_with(Bucket='test-bucket')
            mock_client.upload_file.assert_called_once()
            mock_client.list_objects_v2.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
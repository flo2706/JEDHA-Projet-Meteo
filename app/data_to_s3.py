#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import boto3
import logging
from pathlib import Path
from dotenv import load_dotenv
from botocore.exceptions import ClientError, NoCredentialsError

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class S3DataUploader:
    """Classe pour gérer l'upload des données vers S3"""
    
    def __init__(self, bucket_name="my-jedha-bucket", s3_folder="meteo_data"):
        """
        Initialise le client S3 et les paramètres
        
        Args:
            bucket_name (str): Nom du bucket S3
            s3_folder (str): Dossier de destination dans S3
        """
        self.bucket_name = bucket_name
        self.s3_folder = s3_folder
        
        # Récupération des credentials depuis les variables d'environnement
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'eu-north-1')
        
        # Initialisation du client S3
        self.s3_client = self._create_s3_client()
    
    def _create_s3_client(self):
        """Crée le client S3 avec les credentials"""
        try:
            if not self.aws_access_key or not self.aws_secret_key:
                raise ValueError("AWS credentials non trouvés dans les variables d'environnement")
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
            
            logger.info("Client S3 initialisé avec succès")
            return s3_client
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du client S3: {e}")
            raise
    
    def check_bucket_exists(self):
        """Vérifie si le bucket existe"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' existe et est accessible")
            return True
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                logger.error(f"Bucket '{self.bucket_name}' n'existe pas")
            elif error_code == 403:
                logger.error(f"Accès refusé au bucket '{self.bucket_name}'")
            else:
                logger.error(f"Erreur lors de la vérification du bucket: {e}")
            return False
    
    def upload_file(self, local_file_path, s3_key=None):
        """
        Upload un fichier vers S3
        
        Args:
            local_file_path (str): Chemin vers le fichier local
            s3_key (str): Clé S3 de destination (optionnel)
        
        Returns:
            bool: True si l'upload a réussi, False sinon
        """
        try:
            local_path = Path(local_file_path)
            
            if not local_path.exists():
                logger.error(f"Fichier local non trouvé: {local_file_path}")
                return False
            
            # Génération de la clé S3 si non fournie
            if s3_key is None:
                s3_key = f"{self.s3_folder}/{local_path.name}"
            
            logger.info(f"Upload en cours: {local_file_path} -> s3://{self.bucket_name}/{s3_key}")
            
            # Upload du fichier
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key
            )
            
            logger.info(f"Upload réussi: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Fichier non trouvé: {local_file_path}")
            return False
        except NoCredentialsError:
            logger.error("Credentials AWS non valides")
            return False
        except ClientError as e:
            logger.error(f"Erreur S3 lors de l'upload: {e}")
            return False
        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'upload: {e}")
            return False
    
    def upload_directory(self, local_directory):
        """
        Upload tous les fichiers d'un répertoire vers S3
        
        Args:
            local_directory (str): Chemin vers le répertoire local
        
        Returns:
            tuple: (nombre de fichiers uploadés, nombre total de fichiers)
        """
        local_dir = Path(local_directory)
        
        if not local_dir.exists() or not local_dir.is_dir():
            logger.error(f"Répertoire non trouvé: {local_directory}")
            return 0, 0
        
        files = list(local_dir.glob('*'))
        files = [f for f in files if f.is_file()]  # Seulement les fichiers
        
        if not files:
            logger.warning(f"Aucun fichier trouvé dans: {local_directory}")
            return 0, 0
        
        logger.info(f"Upload de {len(files)} fichier(s) depuis {local_directory}")
        
        uploaded_count = 0
        for file_path in files:
            if self.upload_file(str(file_path)):
                uploaded_count += 1
        
        logger.info(f"Upload terminé: {uploaded_count}/{len(files)} fichiers uploadés")
        return uploaded_count, len(files)
    
    def list_s3_objects(self):
        """Liste les objets dans le dossier S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.s3_folder + "/"
            )
            
            if 'Contents' in response:
                logger.info(f"Objets dans s3://{self.bucket_name}/{self.s3_folder}/:")
                for obj in response['Contents']:
                    logger.info(f"  - {obj['Key']} ({obj['Size']} bytes)")
            else:
                logger.info(f"Aucun objet trouvé dans s3://{self.bucket_name}/{self.s3_folder}/")
                
        except ClientError as e:
            logger.error(f"Erreur lors de la liste des objets S3: {e}")


def main():
    """Fonction principale pour uploader les données"""
    
    # Configuration
    DATA_DIRECTORY = "data"  # Répertoire relatif depuis la racine du projet
    
    # Recherche du répertoire data depuis le répertoire courant
    current_dir = Path(__file__).parent.parent  # Remonte d'un niveau depuis app/
    data_path = current_dir / DATA_DIRECTORY
    
    logger.info(f"Répertoire de données: {data_path}")
    
    try:
        # Initialisation de l'uploader S3
        uploader = S3DataUploader()
        
        # Vérification du bucket
        if not uploader.check_bucket_exists():
            logger.error("Impossible de continuer sans accès au bucket S3")
            return False
        
        # Upload des données
        uploaded, total = uploader.upload_directory(str(data_path))
        
        if uploaded == total and total > 0:
            logger.info("✅ Tous les fichiers ont été uploadés avec succès!")
            
            # Affichage du contenu S3
            uploader.list_s3_objects()
            return True
        else:
            logger.warning(f"⚠️ Upload partiel: {uploaded}/{total} fichiers uploadés")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du processus d'upload: {e}")
        return False


if __name__ == "__main__":
    # Variables d'environnement requises
    required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    
    print("🚀 Upload des données météo vers S3")
    print("=" * 50)
    
    # Vérification des variables d'environnement
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Variables d'environnement manquantes: {', '.join(missing_vars)}")
        print("\nVeuillez définir ces variables dans votre fichier .env ou votre environnement:")
        print("AWS_ACCESS_KEY_ID=votre_access_key")
        print("AWS_SECRET_ACCESS_KEY=votre_secret_key")
        print("AWS_DEFAULT_REGION=us-east-1  # optionnel")
        exit(1)
    
    # Lancement de l'upload
    success = main()
    
    if success:
        print("\n✅ Upload terminé avec succès!")
    else:
        print("\n❌ Erreur lors de l'upload")
        exit(1)
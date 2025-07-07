pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'projet-meteo'
    }
    stages {
        stage('Debug') {
            steps {
                sh 'pwd'
                sh 'ls -l'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t ${DOCKER_IMAGE} .'
            }
        }

        stage('Run Docker Container') {
            steps {
                 sh 'docker run ${DOCKER_IMAGE}'
            }
        }
    }

    post {
        success {
            echo 'JEDHA-Projet-Meteo sucessfully build!'
            // Optionally send notification (Slack/Email)
        }
        failure {
            echo 'JEDHA-Projet-Meteo failed.'
            // Optionally send notification (Slack/Email)
        }
    }
}

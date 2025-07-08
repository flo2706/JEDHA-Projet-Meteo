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

        stage('Install & Run Unit Tests') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    PYTHONPATH=. pytest tests/ --junitxml=unit-tests.xml --html=report.html --self-contained-html
                '''
            }
            post {
                always {
                    junit 'unit-tests.xml'
                    publishHTML([ 
                        reportName: 'Test Report',
                        reportDir: '.',
                        reportFiles: 'report.html',
                        keepAll: true,
                        alwaysLinkToLastBuild: true,
                        allowMissing: false
                    ])
                }
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
            echo '✅ JEDHA-Projet-Meteo successfully built and tested!'
        }
        failure {
            echo '❌ JEDHA-Projet-Meteo failed.'
        }
    }
}

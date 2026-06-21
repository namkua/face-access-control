pipeline {
    agent any
    
    environment {
        // DockerHub configurations
        DOCKERHUB_USER = 'namkua'
        DOCKERHUB_CREDS = 'dockerhub'
        IMAGE_NAME = 'face-access-control'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_SHORT = sh(
                        script: "git rev-parse --short HEAD",
                        returnStdout: true
                    ).trim()
                    env.IMAGE_TAG = "${env.GIT_COMMIT_SHORT}"
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    sh """
                        docker build -f api/Dockerfile \
                                     -t ${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG} \
                                     -t ${DOCKERHUB_USER}/${IMAGE_NAME}:latest \
                                     .
                    """
                }
            }
        }
        
        stage('Unit Tests') {
            steps {
                sh """
                    # Run tests inside the Docker container to leverage cached dependencies
                    # Mount workspace so that test results (xml/html) are saved back to Jenkins host
                    docker run --rm --user root -v \${WORKSPACE}:/app -w /app ${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG} \\
                        pytest tests/unit/ -v --cov=api --cov-report=html --cov-report=xml --junitxml=test-results/results.xml
                """
            }
            post {
                always {
                    junit 'test-results/*.xml'
                    publishHTML([
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }
            }
        }
        

        stage('Integration Tests') {
            steps {
                sh """
                    # 1. Start dependencies (Postgres, Minio, Redis, Kafka)
                    docker-compose up -d postgres minio redis zookeeper kafka
                    
                    # 2. Wait for Postgres to be ready
                    echo "Waiting for PostgreSQL to start..."
                    sleep 15
                    
                    # 3. Create test_db and test user for integration tests
                    docker exec face-recognition-postgres psql -U admin -d postgres -c "CREATE DATABASE test_db;" || true
                    docker exec face-recognition-postgres psql -U admin -d postgres -c "CREATE USER test WITH PASSWORD 'test';" || true
                    docker exec face-recognition-postgres psql -U admin -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE test_db TO test;" || true
                    docker exec face-recognition-postgres psql -U admin -d test_db -c "GRANT ALL ON SCHEMA public TO test;" || true
                    
                    # 4. Run Integration Tests in the docker-compose network
                    docker run --rm --user root \\
                        --network face-recognition-mlops_face-recognition-network \\
                        -v \${WORKSPACE}:/app -w /app \\
                        -e MINIO_ENDPOINT=minio:9000 \\
                        -e TEST_DATABASE_URL=postgresql+asyncpg://test:test@postgres:5432/test_db \\
                        -e REDIS_HOST=redis \\
                        -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \\
                        -e JAEGER_AGENT_HOST=jaeger \\
                        ${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG} \\
                        pytest tests/integration/ -v
                """
            }
        }

        
        stage('Push to DockerHub') {
            when {
                branch 'main'
            }
            steps {
                withCredentials([usernamePassword(credentialsId: "${DOCKERHUB_CREDS}", passwordVariable: 'DOCKER_PASS', usernameVariable: 'DOCKER_USER')]) {
                    sh """
                        echo \$DOCKER_PASS | docker login -u \$DOCKER_USER --password-stdin
                        docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}
                        docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:latest
                    """
                }
            }
        }
    }
    
    post {
        always {
            sh '''
                docker-compose down -v || true
            '''
            deleteDir()
        }
        success {
            echo 'Pipeline succeeded!'
            // Send notification
        }
        failure {
            echo 'Pipeline failed!'
            // Send alert
        }
    }
}

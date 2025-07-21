pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
metadata:
  labels:
    app: crowd-sentiment-music-generator-builder
spec:
  containers:
  - name: python
    image: python:3.12-slim
    command:
    - cat
    tty: true
  - name: docker
    image: docker:20.10.16-dind
    command:
    - cat
    tty: true
    privileged: true
    volumeMounts:
    - name: docker-sock
      mountPath: /var/run/docker.sock
  - name: kubectl
    image: bitnami/kubectl:latest
    command:
    - cat
    tty: true
  volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
"""
        }
    }
    
    environment {
        DOCKER_REGISTRY = 'your-registry.example.com'
        IMAGE_NAME = 'crowd-sentiment-music-generator'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        KUBECONFIG = credentials('kubeconfig')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Install Dependencies') {
            steps {
                container('python') {
                    sh 'pip install -e .[dev,test]'
                }
            }
        }
        
        stage('Lint') {
            steps {
                container('python') {
                    sh 'ruff check .'
                    sh 'ruff format --check .'
                    sh 'mypy src tests'
                }
            }
        }
        
        stage('Unit Tests') {
            steps {
                container('python') {
                    sh 'pytest tests/unit -v'
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                container('docker') {
                    sh """
                    docker build -t ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .
                    docker tag ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest
                    """
                }
            }
        }
        
        stage('Push Docker Image') {
            steps {
                container('docker') {
                    withCredentials([string(credentialsId: 'docker-registry-token', variable: 'DOCKER_TOKEN')]) {
                        sh """
                        echo \${DOCKER_TOKEN} | docker login ${DOCKER_REGISTRY} -u jenkins --password-stdin
                        docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
                        docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest
                        """
                    }
                }
            }
        }
        
        stage('Deploy to Development') {
            when {
                branch 'develop'
            }
            steps {
                container('kubectl') {
                    sh """
                    export KUBECONFIG=\${KUBECONFIG}
                    
                    # Apply Kubernetes manifests with environment-specific values
                    envsubst < k8s/namespace.yaml | kubectl apply -f -
                    envsubst < k8s/deployment.yaml | kubectl apply -f -
                    envsubst < k8s/service.yaml | kubectl apply -f -
                    envsubst < k8s/hpa.yaml | kubectl apply -f -
                    
                    # Wait for deployment to be ready
                    kubectl rollout status deployment/crowd-sentiment-music-generator -n crowd-sentiment-music-generator --timeout=300s
                    """
                }
            }
        }
        
        stage('Integration Tests') {
            when {
                branch 'develop'
            }
            steps {
                container('python') {
                    sh """
                    # Set environment variables for integration tests
                    export API_BASE_URL=http://crowd-sentiment-music-generator.crowd-sentiment-music-generator.svc.cluster.local
                    
                    # Run integration tests
                    pytest tests/integration -v
                    """
                }
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                container('kubectl') {
                    sh """
                    export KUBECONFIG=\${KUBECONFIG}
                    
                    # Apply Kubernetes manifests with production-specific values
                    export KUBERNETES_NAMESPACE=crowd-sentiment-music-generator-prod
                    export MIN_INSTANCES=3
                    export MAX_INSTANCES=20
                    
                    envsubst < k8s/namespace.yaml | kubectl apply -f -
                    envsubst < k8s/deployment.yaml | kubectl apply -f -
                    envsubst < k8s/service.yaml | kubectl apply -f -
                    envsubst < k8s/hpa.yaml | kubectl apply -f -
                    
                    # Wait for deployment to be ready
                    kubectl rollout status deployment/crowd-sentiment-music-generator -n \${KUBERNETES_NAMESPACE} --timeout=300s
                    """
                }
            }
        }
    }
    
    post {
        always {
            junit 'test-results/*.xml'
        }
        success {
            echo 'Build and deployment successful!'
        }
        failure {
            echo 'Build or deployment failed!'
        }
    }
}
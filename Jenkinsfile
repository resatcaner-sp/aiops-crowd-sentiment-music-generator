library "devops-jenkins-shared-libs@1.25.1"

pipeline {
    agent {
        label 'c512m1g && euw1-prod'
    }

    options {
        buildDiscarder logRotator(numToKeepStr: '5')
        timeout(time: 20, unit: 'MINUTES')
    }

    environment {
        PYTHON_VERSION = "3.12-slim"
        POETRY_VERSION = "2.0.1"
    }

    stages {
        stage('Initialization') {
            when {
                branch 'main'
            }
            steps {
                echo "Running on the main branch"
            }
        }
        stage('Test') {
            steps {
                poetry 'install'
                poetry 'run python -m pytest --cov --cov-report xml --junitxml=report.xml'
                poetry 'run ruff check --output-file ruff_report.json --output-format json --exit-zero'
                poetry 'run mypy src --output json > mypy_report.json 2>/dev/null || true'
                junit 'report.xml'
            }
        }

        stage('Sonar') {
            steps {
                sonarScanner ''
            }
        }


        stage ('Publish') {
            when {
                buildingTag() // builds should only be published from tags
            }

            steps {
                // verify the tag and version in pyproject.toml match
                poetry "version -s | grep -qFx '${TAG_NAME}' "
                poetry 'publish --build --repository hosted'
            }
        }
    }

    post {
        always {
            deleteDir()
        }
    }
}

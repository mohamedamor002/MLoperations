# Variables
PYTHON=python
PIP=pip
VENV=test
REQ=requirements.txt
TRAINED_MODEL=trained_model.joblib
SCALER=scaler.joblib
PREPROCESSED_DATA=preprocessed_data.joblib
DATA_FILE=data.csv
TEST_DATA_FILE=testData.csv
TARGET_COLUMN=Churn

# Docker Variables
DOCKER_IMAGE=mamor02/fastapi-mlflow-app
DOCKER_TAG=latest

# MLflow Variables
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_ARTIFACT_LOCATION=./mlruns

# Create a virtual environment and install dependencies
install:
	@echo "Creating virtual environment..."
	${PYTHON} -m venv ${VENV}
	@echo "Activating virtual environment and installing dependencies..."
	${VENV}/bin/${PIP} install -r ${REQ}

# Format and lint the code
lint:
	@echo "Checking code formatting and quality..."
	${VENV}/bin/black --check .
	${VENV}/bin/flake8 .

# Prepare the data
prepare:
	@echo "Preprocessing data..."
	${VENV}/bin/python model_pipeline.py preprocess --file_path ${DATA_FILE} --target_column ${TARGET_COLUMN}

# Train the model
train:
	@echo "Training the model..."
	${VENV}/bin/python model_pipeline.py train --file_path ${DATA_FILE} --target_column ${TARGET_COLUMN}

# Test the model on test data
test:
	@echo "Testing the model..."
	${VENV}/bin/python model_pipeline.py test --file_path ${TEST_DATA_FILE} --target_column ${TARGET_COLUMN}

# Run both prepare, train, and test
all: prepare train test

# Clean generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f ${TRAINED_MODEL} ${SCALER} ${PREPROCESSED_DATA}

# Run tests (if applicable)
unit-test:
	@echo "Running unit tests..."
	${VENV}/bin/pytest tests/

# Run the API and display Swagger UI URL
run-api:
	@echo "Starting API..."
	${VENV}/bin/uvicorn app:app --reload &
	@echo "Waiting for the API to start..."
	@sleep 2
	@echo "API started. Open Swagger UI at: http://127.0.0.1:8000/docs"

# Run the Flask application
run-flask:
	@echo "Starting Flask application..."
	${VENV}/bin/python app_flask.py

# Monitor changes and run make all
monitor:
	@echo "Starting file monitoring..."
	./monitor_changes.sh

# Run MLflow UI
run-mlflow:
	@echo "Starting MLflow UI..."
	${VENV}/bin/mlflow ui --backend-store-uri ${MLFLOW_TRACKING_URI} --host 0.0.0.0 --port 5006 --default-artifact-root ${MLFLOW_ARTIFACT_LOCATION}

# Docker Commands

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .

# Run Docker container
docker-run:
	@echo "Running Docker container..."
	docker run -p 8080:8000 ${DOCKER_IMAGE}:${DOCKER_TAG} 

# Push Docker image to Docker Hub
docker-push:
	@echo "Pushing Docker image to Docker Hub..."
	docker push ${DOCKER_IMAGE}:${DOCKER_TAG}

# Clean Docker images and containers
docker-clean:
	@echo "Cleaning up Docker images and containers..."
	docker system prune -f

.PHONY: install lint prepare train test all clean unit-test run-api run-flask monitor run-mlflow docker-build docker-run docker-push docker-clean
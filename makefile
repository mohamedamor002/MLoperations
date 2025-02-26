# Variables
PYTHON=python
PIP=pip
VENV=test
REQ=requirements.txt
TRAINED_MODEL=trained_model.joblib
SCALER=scaler.joblib
PREPROCESSED_DATA=preprocessed_data.joblib
DATA_FILE=/home/amor/ml_project/data.csv
TARGET_COLUMN=Churn

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
	${VENV}/bin/python model_pipeline.py preprocess

# Train the model
train:
	@echo "Training the model..."
	${VENV}/bin/python model_pipeline.py train

# Run both prepare and train
all: prepare train

# Clean generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f ${TRAINED_MODEL} ${SCALER} ${PREPROCESSED_DATA}

# Run tests (if applicable)
test:
	@echo "Running tests..."
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
	${VENV}/bin/mlflow ui --host 127.0.0.1 --port 5004

.PHONY: install lint prepare train all clean test run-api run-flask monitor run-mlflow
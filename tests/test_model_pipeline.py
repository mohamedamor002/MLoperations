import sys
import os
import pytest
import pandas as pd
import joblib

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from model_pipeline
from model_pipeline import load_data, preprocess_data, train_model

# Fixture to load data once and reuse in multiple tests
@pytest.fixture
def sample_data():
    file_path = "/home/amor/ml_project/churn-bigml-80.csv"  # Update with your actual file path
    return load_data(file_path)

# Test data loading
def test_load_data(sample_data):
    """
    Test that data is loaded correctly and is not empty.
    """
    assert isinstance(sample_data, pd.DataFrame), "Loaded data is not a DataFrame"
    assert not sample_data.empty, "Loaded data is empty"

# Test data preprocessing
def test_preprocess_data(sample_data):
    """
    Test that data preprocessing works as expected.
    """
    target_column = "Churn"  # Update with your actual target column
    X, y, scaler = preprocess_data(sample_data, target_column)

    # Check that features and target are not empty
    assert not X.empty, "Features (X) are empty after preprocessing"
    assert not y.empty, "Target (y) is empty after preprocessing"

    # Check that the scaler is fitted
    assert hasattr(scaler, "mean_"), "Scaler is not fitted"

# Test model training
def test_train_model(sample_data):
    """
    Test that the model can be trained and returns valid outputs.
    """
    target_column = "Churn"  # Update with your actual target column
    X, y, scaler = preprocess_data(sample_data, target_column)

    # Train the model
    model, accuracy, report, X_test = train_model(X, y)  # Unpack four values

    # Check that the model is trained
    assert hasattr(model, "fit"), "Model is not trained"
    assert isinstance(accuracy, float), "Accuracy is not a float"
    assert accuracy >= 0 and accuracy <= 1, "Accuracy is out of valid range (0-1)"
    assert isinstance(report, str), "Classification report is not a string"
    assert isinstance(X_test, pd.DataFrame), "X_test is not a DataFrame"

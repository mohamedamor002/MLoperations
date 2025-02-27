import logging
import subprocess

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()          # Log to the console
    ]
)

# Load the trained model
MODEL_PATH = "trained_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model from {MODEL_PATH}: {e}")
    raise RuntimeError(f"Error loading model: {e}")

# Initialize FastAPI app
app = FastAPI()

# Define request body for prediction


class PredictionInput(BaseModel):
    account_length: float
    # Assuming this is a categorical variable (e.g., "yes" or "no")
    international_plan: str
    # Assuming this is a categorical variable (e.g., "yes" or "no")
    voice_mail_plan: str
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_eve_minutes: float
    total_eve_calls: int
    total_night_minutes: float
    total_night_calls: int
    total_intl_minutes: float
    total_intl_calls: int
    customer_service_calls: int

# Define request body for retraining


class RetrainParams(BaseModel):
    file_path: str
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    kernel: str = 'rbf'

# Define prediction endpoint


@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert input to numpy array
        input_array = np.array([
            input_data.account_length,
            1 if input_data.international_plan.lower() == "yes" else 0,
            1 if input_data.voice_mail_plan.lower() == "yes" else 0,
            input_data.number_vmail_messages,
            input_data.total_day_minutes,
            input_data.total_day_calls,
            input_data.total_eve_minutes,
            input_data.total_eve_calls,
            input_data.total_night_minutes,
            input_data.total_night_calls,
            input_data.total_intl_minutes,
            input_data.total_intl_calls,
            input_data.customer_service_calls
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)
        logging.info(f"Prediction successful: {prediction}")
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Define retrain endpoint


@app.post("/retrain")
def retrain_model(params: RetrainParams):
    try:
        # Execute the training script with new hyperparameters
        subprocess.run(
            ["python", "model_pipeline.py", "train",
             "--file_path", params.file_path,
             "--target_column", params.target_column,
             "--test_size", str(params.test_size),
             "--random_state", str(params.random_state),
             "--kernel", params.kernel],
            check=True
        )
        logging.info("Model retrained successfully.")
        return {"message": "Model retrained successfully"}
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error during retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with: uvicorn app:app --reload

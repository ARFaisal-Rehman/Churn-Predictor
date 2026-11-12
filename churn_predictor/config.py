import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    DATA_PATH = Path(os.getenv("CHURN_DATA_PATH", BASE_DIR / "data.csv"))
    ARTIFACTS_DIR = Path(os.getenv("CHURN_ARTIFACTS_DIR", BASE_DIR / "artifacts"))
    MODEL_PATH = ARTIFACTS_DIR / "model_bundle.joblib"
    ANALYSIS_PATH = ARTIFACTS_DIR / "analysis.json"
    PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
    TRAINING_PATH = ARTIFACTS_DIR / "training_dataset.csv"
    PREDICTION_WINDOW_DAYS = int(os.getenv("PREDICTION_WINDOW_DAYS", "60"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
    TEST_SIZE = float(os.getenv("TEST_SIZE", "0.25"))

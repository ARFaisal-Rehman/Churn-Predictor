import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")  # no sessions or forms use it; set it if you add any
    DATA_PATH = Path(os.getenv("CHURN_DATA_PATH", BASE_DIR / "data" / "online_retail_ii.csv"))
    ARTIFACTS_DIR = Path(os.getenv("CHURN_ARTIFACTS_DIR", BASE_DIR / "artifacts"))
    MODEL_PATH = ARTIFACTS_DIR / "model_bundle.joblib"
    ANALYSIS_PATH = ARTIFACTS_DIR / "analysis.json"
    PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
    METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
    PREDICTION_WINDOW_DAYS = int(os.getenv("PREDICTION_WINDOW_DAYS", "60"))
    ACTIVE_LOOKBACK_DAYS = int(os.getenv("ACTIVE_LOOKBACK_DAYS", "365"))
    SNAPSHOT_STEP_DAYS = int(os.getenv("SNAPSHOT_STEP_DAYS", "30"))
    USE_SEASON_FEATURES = os.getenv("USE_SEASON_FEATURES", "0") == "1"
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

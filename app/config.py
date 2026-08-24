"""Application configuration and transparent drought-risk constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BACKEND_DIR / "app"
DATA_DIR = APP_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
DATASET_ROOT = BACKEND_DIR / "data" / "patches"
ML_DIR = BACKEND_DIR / "ml"
ML_WEIGHTS_DIR = ML_DIR / "weights"

load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime values which may be overridden in backend/.env."""

    app_name: str = os.getenv("APP_NAME", "AI Drought Monitoring Platform")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    frontend_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    )


settings = Settings()

# Risk bands required by the MVP.
RISK_LOW_MAX = 0.40
RISK_MEDIUM_MAX = 0.70

# Expected local-MVP ranges. Values outside a range are clipped before scoring.
NDVI_RANGE = (0.10, 0.85)
NDMI_RANGE = (-0.40, 0.50)
TEMPERATURE_C_RANGE = (5.0, 45.0)
PRECIPITATION_MM_RANGE = (0.0, 35.0)
SOIL_MOISTURE_RANGE = (0.05, 0.55)

# Every factor is normalized to 0..1, then combined using these documented weights.
RISK_WEIGHTS = {
    "ndvi": 0.25,
    "ndmi": 0.20,
    "temperature_c": 0.20,
    "precipitation_mm": 0.15,
    "soil_moisture": 0.20,
}
CONTRIBUTOR_FACTOR_THRESHOLD = 0.60

MODEL_RANDOM_SEED = 42
MODEL_TEST_SIZE = 0.25
MODEL_ESTIMATORS = 200

SAMPLE_OBSERVATIONS_PATH = DATA_DIR / "sample_observations.csv"
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
MODEL_PATH = MODELS_DIR / "drought_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

PATCH_SIZE = 256
PATCH_CHANNELS = ("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI")
PATCH_SCALE_METERS = 10
GEE_GROWING_SEASON_START_MONTH = 5
GEE_GROWING_SEASON_END_MONTH = 8
VCI_DROUGHT_THRESHOLD_PERCENT = 35.0
SEGMENTATION_MODEL_PATH = ML_WEIGHTS_DIR / "best_model.pth"

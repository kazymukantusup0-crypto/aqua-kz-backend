"""Load and use the persisted local drought classifier."""

from __future__ import annotations

import joblib
import pandas as pd

from app.config import MODEL_PATH
from app.ml.features import FEATURE_COLUMNS
from app.schemas import ObservationInput, PredictionResponse


class ModelNotTrainedError(RuntimeError):
    """Raised when a prediction is requested before local training."""


def predict_drought(observation: ObservationInput) -> PredictionResponse:
    """Produce a probability and binary educational classification from the saved pipeline."""
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            "Model has not been trained. Call POST /api/train-model before POST /api/predict."
        )
    model = joblib.load(MODEL_PATH)
    feature_frame = pd.DataFrame(
        [{name: getattr(observation, name) for name in FEATURE_COLUMNS}]
    )
    label = int(model.predict(feature_frame)[0])
    probability = float(model.predict_proba(feature_frame)[0][1])
    return PredictionResponse(
        drought_label=label,
        prediction="drought_prone" if label else "not_drought_prone",
        drought_probability=round(probability, 4),
    )

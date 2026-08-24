"""Train and persist the educational synthetic-data classifier."""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import (
    METRICS_PATH,
    MODEL_ESTIMATORS,
    MODEL_PATH,
    MODEL_RANDOM_SEED,
    MODEL_TEST_SIZE,
    TRAINING_DATA_PATH,
)
from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from app.schemas import TrainingMetrics


def train_and_save_model() -> TrainingMetrics:
    """Train a reproducible pipeline on explicitly synthetic CSV data."""
    # The leading CSV comment documents that this educational dataset is synthetic.
    data = pd.read_csv(TRAINING_DATA_PATH, comment="#")
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {sorted(missing_columns)}")
    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("drought_label must contain only 0 or 1")

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS],
        data[TARGET_COLUMN],
        test_size=MODEL_TEST_SIZE,
        random_state=MODEL_RANDOM_SEED,
        stratify=data[TARGET_COLUMN],
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=MODEL_ESTIMATORS,
                    random_state=MODEL_RANDOM_SEED,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = TrainingMetrics(
        status="trained",
        accuracy=round(float(accuracy_score(y_test, predictions)), 4),
        precision=round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        recall=round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        f1=round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        training_rows=len(data),
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    return metrics


def load_saved_metrics() -> TrainingMetrics:
    """Return persisted metrics, or an explicit untrained status."""
    if not METRICS_PATH.exists():
        return TrainingMetrics(status="not_trained")
    return TrainingMetrics.model_validate(json.loads(METRICS_PATH.read_text(encoding="utf-8")))

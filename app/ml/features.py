"""Shared feature order for model training and prediction."""

FEATURE_COLUMNS = [
    "ndvi",
    "ndmi",
    "temperature_c",
    "precipitation_mm",
    "soil_moisture",
]
TARGET_COLUMN = "drought_label"

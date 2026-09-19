from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.data.cleaning import clean_data
from src.data.load_data import load_test_data
from src.features.build_features import build_features
from src.models.classification import predict_invalid_probability
from src.summary import generate_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_PATH = SUBMISSION_DIR / "0xPOWER.csv"
SUMMARY_PATH = SUBMISSION_DIR / "summary.json"
REQUIRED_ARTIFACTS = [
    "classifier.joblib", "regressor.joblib", "imputer.joblib",
    "feature_columns.joblib", "invalid_threshold.joblib",
]


def load_model_artifacts() -> tuple:
    missing = [name for name in REQUIRED_ARTIFACTS if not (MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Model artifacts are missing from {MODEL_DIR}: {', '.join(missing)}. Run `python src/train.py` first."
        )
    return tuple(joblib.load(MODEL_DIR / name) for name in REQUIRED_ARTIFACTS)


def validate_submission(submission: pd.DataFrame, test_df: pd.DataFrame) -> None:
    expected_columns = ["Test_ID", "Predicted_Reference_Parameter", "Validity_Label"]
    if list(submission.columns) != expected_columns or len(submission) != len(test_df):
        raise ValueError("Submission columns or row count are invalid.")
    if not submission["Test_ID"].equals(test_df["Test_ID"].reset_index(drop=True)):
        raise ValueError("Submission Test_ID order does not match Test_Data.")
    if submission["Test_ID"].duplicated().any() or submission["Test_ID"].isna().any():
        raise ValueError("Test_ID values must be present and unique.")
    predictions = submission["Predicted_Reference_Parameter"].to_numpy()
    if pd.isna(predictions).any() or not np.isfinite(predictions).all():
        raise ValueError("Reference predictions must be finite and non-missing.")
    if not submission["Validity_Label"].isin(["Valid", "Invalid"]).all():
        raise ValueError("Validity labels must be Valid or Invalid.")


def generate_predictions() -> tuple[pd.DataFrame, dict]:
    test_df = clean_data(load_test_data(), training=False)
    classifier, regressor, imputer, feature_columns, threshold = load_model_artifacts()
    features = build_features(test_df)
    if list(features.columns) != list(feature_columns):
        raise ValueError("Inference feature columns do not match the trained feature order.")
    transformed = imputer.transform(features)
    reference_predictions = regressor.predict(transformed)
    invalid_probabilities = predict_invalid_probability(classifier, transformed)
    labels = np.where(invalid_probabilities >= threshold, "Invalid", "Valid")
    submission = pd.DataFrame({
        "Test_ID": test_df["Test_ID"].reset_index(drop=True),
        "Predicted_Reference_Parameter": np.round(reference_predictions, 4),
        "Validity_Label": labels,
    })
    validate_submission(submission, test_df.reset_index(drop=True))
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    summary = generate_summary(submission, invalid_probabilities, SUMMARY_PATH)
    return submission, summary


if __name__ == "__main__":
    generate_predictions()

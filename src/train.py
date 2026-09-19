from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.impute import SimpleImputer

from src.data.cleaning import clean_data
from src.data.load_data import load_training_data
from src.features.build_features import FEATURE_COLUMNS, build_features
from src.models.classification import INVALID_THRESHOLD, create_classifier
from src.models.regression import create_regressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


def train_and_save_models(train_df=None) -> tuple:
    """Fit final models and persist all inference-required artifacts."""
    if train_df is None:
        train_df = load_training_data()
    train_df = clean_data(train_df, training=True)
    features = build_features(train_df)
    target_class = (train_df["Validity_Label"] == "Invalid").astype(int)
    target_regression = train_df["Reference_Parameter"]

    imputer = SimpleImputer(strategy="median")
    transformed = imputer.fit_transform(features)
    classifier = create_classifier().fit(transformed, target_class)
    regressor = create_regressor().fit(transformed, target_regression)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_DIR / "classifier.joblib")
    joblib.dump(regressor, MODEL_DIR / "regressor.joblib")
    joblib.dump(imputer, MODEL_DIR / "imputer.joblib")
    joblib.dump(FEATURE_COLUMNS, MODEL_DIR / "feature_columns.joblib")
    joblib.dump(INVALID_THRESHOLD, MODEL_DIR / "invalid_threshold.joblib")
    return classifier, regressor, imputer, FEATURE_COLUMNS, INVALID_THRESHOLD


def main() -> None:
    train_and_save_models()
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()

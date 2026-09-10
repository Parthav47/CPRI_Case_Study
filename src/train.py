from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import build_feature_matrix


def train_models(train_df: pd.DataFrame) -> tuple[RandomForestClassifier, RandomForestRegressor, SimpleImputer, list[str]]:
    """Train the final classification and regression models."""
    X_train, _, feature_cols = build_feature_matrix(train_df, train_df)

    y_class = (train_df["Validity_Label"] == "Invalid").astype(int)
    y_reg = train_df["Reference_Parameter"]

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)

    classifier = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        max_features=0.5,
        min_samples_leaf=1,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    regressor = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    classifier.fit(X_train_imputed, y_class)
    regressor.fit(X_train_imputed, y_reg)

    return classifier, regressor, imputer, feature_cols


def evaluate_training(train_df: pd.DataFrame) -> dict:
    """Quick training-time metrics for sanity checking."""
    classifier, regressor, _, _ = train_models(train_df)

    X_train, _, _ = build_feature_matrix(train_df, train_df)
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)

    y_class = (train_df["Validity_Label"] == "Invalid").astype(int)
    y_reg = train_df["Reference_Parameter"]

    class_prob = classifier.predict_proba(X_train_imputed)[:, 1]
    class_pred = (class_prob >= 0.415).astype(int)

    reg_pred = regressor.predict(X_train_imputed)

    metrics = {
        "mae": float(mean_absolute_error(y_reg, reg_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_reg, reg_pred))),
        "r2": float(r2_score(y_reg, reg_pred)),
        "class_balance": {
            "valid": int((1 - class_pred).sum()),
            "invalid": int(class_pred.sum()),
        },
    }

    return metrics


def main() -> None:
    train_path = Path("../data/processed/train_relationship_features.csv")
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_df = pd.read_csv(train_path)
    metrics = evaluate_training(train_df)

    output = Path("../reports")
    output.mkdir(exist_ok=True, parents=True)
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

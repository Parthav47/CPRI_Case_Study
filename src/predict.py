from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import build_feature_matrix
from train import train_models


def predict(test_df: pd.DataFrame, model_path: str | None = None) -> pd.DataFrame:
    """Generate model predictions for the test set."""
    train_df = pd.read_csv("../data/processed/train_relationship_features.csv")
    classifier, regressor, imputer, feature_cols = train_models(train_df)

    X_train, X_test, _ = build_feature_matrix(train_df, test_df)
    X_test_imputed = imputer.transform(X_test)

    pred_ref = regressor.predict(X_test_imputed)
    pred_invalid_prob = classifier.predict_proba(X_test_imputed)[:, 1]
    pred_validity = np.where(pred_invalid_prob >= 0.415, "Invalid", "Valid")

    submission = pd.DataFrame(
        {
            "Test_ID": test_df["Test_ID"],
            "Predicted_Reference_Parameter": pred_ref,
            "Validity_Label": pred_validity,
        }
    )

    return submission


def main() -> None:
    test_path = Path("../data/processed/test_relationship_features.csv")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    test_df = pd.read_csv(test_path)
    submission = predict(test_df)

    out_dir = Path("../submissions")
    out_dir.mkdir(exist_ok=True, parents=True)
    submission.to_csv(out_dir / "TEAMNAME.csv", index=False)

    print(submission.head())


if __name__ == "__main__":
    main()

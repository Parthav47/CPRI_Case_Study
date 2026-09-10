from __future__ import annotations

from typing import Iterable

import pandas as pd


def engineer_sensor_relationship_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple sensor-based features used for validity and reference modeling."""
    out = df.copy()

    sensor_cols = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
    for col in sensor_cols:
        if col not in out.columns:
            continue

    out["S2_S1_residual"] = out.get("Sensor_S2", 0) - out.get("Sensor_S1", 0)
    out["S3_S1_residual"] = out.get("Sensor_S3", 0) - out.get("Sensor_S1", 0)
    out["S3_S2_residual"] = out.get("Sensor_S3", 0) - out.get("Sensor_S2", 0)

    out["Voltage_Current"] = (
        out.get("Applied_Voltage_kV", 0) * out.get("Load_Current_A", 0)
    )

    for col in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype(int)

    return out


def build_feature_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create feature sets for model training and inference."""
    train = engineer_sensor_relationship_features(train_df)
    test = engineer_sensor_relationship_features(test_df)

    base_features = [
        "Applied_Voltage_kV",
        "Load_Current_A",
        "Ambient_Temperature_C",
        "Test_Duration_min",
        "Sensor_S1",
        "Sensor_S2",
        "Sensor_S3",
        "Sensor_S4",
        "S2_S1_residual",
        "S3_S1_residual",
        "S3_S2_residual",
        "Voltage_Current",
        "Sensor_S1_missing",
        "Sensor_S2_missing",
        "Sensor_S3_missing",
        "Sensor_S4_missing",
    ]

    feature_cols = [col for col in base_features if col in train.columns and col in test.columns]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    return X_train, X_test, feature_cols

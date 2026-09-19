"""Canonical inference-safe feature engineering."""

import pandas as pd

FEATURE_COLUMNS = [
	"Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min",
	"Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
	"S2_S1_residual", "S3_S1_residual", "S3_S2_residual",
	"S2_S1_residual_abs", "S3_S1_residual_abs", "S3_S2_residual_abs",
	"S2_minus_S1", "S3_minus_S1", "S3_minus_S2", "Voltage_Current",
	"Sensor_S1_missing", "Sensor_S2_missing", "Sensor_S3_missing", "Sensor_S4_missing",
]
BASE_COLUMNS = FEATURE_COLUMNS[:8]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Build exactly the 22 features used by both training and inference."""
	missing = set(BASE_COLUMNS).difference(df.columns)
	if missing:
		raise ValueError(f"Cannot build features; missing columns: {sorted(missing)}")
	out = df.copy()
	for column in BASE_COLUMNS:
		out[column] = pd.to_numeric(out[column], errors="coerce")
	out["S2_S1_residual"] = out["Sensor_S2"] - out["Sensor_S1"]
	out["S3_S1_residual"] = out["Sensor_S3"] - out["Sensor_S1"]
	out["S3_S2_residual"] = out["Sensor_S3"] - out["Sensor_S2"]
	out["S2_S1_residual_abs"] = out["S2_S1_residual"].abs()
	out["S3_S1_residual_abs"] = out["S3_S1_residual"].abs()
	out["S3_S2_residual_abs"] = out["S3_S2_residual"].abs()
	out["S2_minus_S1"] = out["Sensor_S2"] - out["Sensor_S1"]
	out["S3_minus_S1"] = out["Sensor_S3"] - out["Sensor_S1"]
	out["S3_minus_S2"] = out["Sensor_S3"] - out["Sensor_S2"]
	out["Voltage_Current"] = out["Applied_Voltage_kV"] * out["Load_Current_A"]
	for column in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]:
		out[f"{column}_missing"] = out[column].isna().astype(int)
	return out.loc[:, FEATURE_COLUMNS]

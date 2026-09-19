"""Validation and conservative cleaning utilities."""

import pandas as pd

IDENTIFIER_COLUMN = "Test_ID"
TARGET_COLUMNS = {"Reference_Parameter", "Validity_Label"}
REQUIRED_COLUMNS = {
	IDENTIFIER_COLUMN, "Applied_Voltage_kV", "Load_Current_A",
	"Ambient_Temperature_C", "Test_Duration_min", "Sensor_S1",
	"Sensor_S2", "Sensor_S3", "Sensor_S4",
}


def validate_columns(df: pd.DataFrame, *, training: bool = False) -> None:
	required = REQUIRED_COLUMNS | (TARGET_COLUMNS if training else set())
	missing = required.difference(df.columns)
	if missing:
		raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def clean_data(df: pd.DataFrame, *, training: bool = False) -> pd.DataFrame:
	"""Validate columns and coerce modeled measurements to numeric values."""
	validate_columns(df, training=training)
	cleaned = df.copy()
	cleaned[IDENTIFIER_COLUMN] = cleaned[IDENTIFIER_COLUMN].astype("string")
	numeric_columns = sorted(REQUIRED_COLUMNS - {IDENTIFIER_COLUMN})
	if training:
		numeric_columns.append("Reference_Parameter")
	for column in numeric_columns:
		cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
	if cleaned[IDENTIFIER_COLUMN].isna().any() or cleaned[IDENTIFIER_COLUMN].duplicated().any():
		raise ValueError("Test_ID values must be present and unique.")
	if training and not cleaned["Validity_Label"].isin(["Valid", "Invalid"]).all():
		raise ValueError("Validity_Label must contain only Valid or Invalid.")
	return cleaned

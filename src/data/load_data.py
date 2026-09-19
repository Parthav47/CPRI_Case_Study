"""Load the authorized challenge workbook from project-relative paths."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
EXPECTED_SHEETS = {"Training_Data", "Test_Data"}


def load_workbook(path: Path = DATA_PATH) -> pd.ExcelFile:
	"""Open the workbook after checking that both required sheets exist."""
	if not path.exists():
		raise FileNotFoundError(
			f"Authorized dataset not found at {path}. Place the workbook there before running the pipeline."
		)
	workbook = pd.ExcelFile(path)
	missing_sheets = EXPECTED_SHEETS.difference(workbook.sheet_names)
	if missing_sheets:
		raise ValueError(f"Dataset is missing required sheets: {sorted(missing_sheets)}")
	return workbook


def load_sheet(sheet_name: str, path: Path = DATA_PATH) -> pd.DataFrame:
	"""Load and normalize one required workbook sheet."""
	if sheet_name not in EXPECTED_SHEETS:
		raise ValueError(f"Unsupported sheet: {sheet_name}")
	with load_workbook(path) as workbook:
		frame = workbook.parse(sheet_name=sheet_name)
	frame.columns = frame.columns.astype(str).str.strip()
	return frame


def load_training_data(path: Path = DATA_PATH) -> pd.DataFrame:
	return load_sheet("Training_Data", path)


def load_test_data(path: Path = DATA_PATH) -> pd.DataFrame:
	return load_sheet("Test_Data", path)

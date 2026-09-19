from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_summary(submission_df: pd.DataFrame, invalid_probabilities) -> dict:
    """Build a result summary from predictions and their Invalid probabilities."""
    ranked = pd.DataFrame({
        "Test_ID": submission_df["Test_ID"].to_numpy(),
        "probability": invalid_probabilities,
    }).sort_values("probability", ascending=False, kind="mergesort")
    values = submission_df["Predicted_Reference_Parameter"]
    return {
        "record_count": int(len(submission_df)),
        "abnormal_invalid_count": int((submission_df["Validity_Label"] == "Invalid").sum()),
        "valid_count": int((submission_df["Validity_Label"] == "Valid").sum()),
        "predicted_reference_parameter": {
            "minimum": float(values.min()), "maximum": float(values.max()), "average": float(values.mean()),
        },
        "top_3_test_ids_requiring_highest_attention": ranked["Test_ID"].head(3).tolist(),
        "attention_method": "Ranked by predicted probability of Invalid classification.",
        "explanation": "Records are ranked by predicted Invalid probability. Reference_Parameter is predicted with the final Random Forest regressor.",
    }


def generate_summary(submission_df: pd.DataFrame, invalid_probabilities, output_path: Path) -> dict:
    summary = build_summary(submission_df, invalid_probabilities)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

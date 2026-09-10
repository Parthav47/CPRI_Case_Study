from __future__ import annotations

import json
from pathlib import Path


def build_summary() -> dict:
    """Create a simple summary payload for the project submission."""
    return {
        "team": "TEAMNAME",
        "model_type": "Random Forest",
        "status": "ready_for_submission",
        "notes": "Baseline feature engineering and validation workflow for the CPRI Hackathon project.",
    }


def main() -> None:
    summary = build_summary()
    out_dir = Path("../submissions")
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

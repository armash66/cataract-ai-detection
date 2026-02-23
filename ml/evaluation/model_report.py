"""Generate model comparison report."""
import json
from pathlib import Path


def save_report(report: dict, out_file: str = "ml/experiments/model_report.json"):
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved report to", path)

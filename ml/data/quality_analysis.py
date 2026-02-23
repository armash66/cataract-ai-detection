"""Aggregate quality reports from phase1 cleaning runs."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    phase1 = root / "ml" / "experiments" / "phase1"
    out = root / "ml" / "data" / "dataset_summary.json"

    report_files = sorted(phase1.glob("quality_report_*.json"))
    if not report_files:
        print("No quality reports found in", phase1)
        return

    totals = {"processed": 0, "saved": 0, "rejected": 0}
    reject_reasons = {"missing_file": 0, "blurry": 0, "too_dark": 0, "too_bright": 0}

    for rf in report_files:
        data = json.loads(rf.read_text(encoding="utf-8"))
        totals["processed"] += int(data.get("processed", 0))
        totals["saved"] += int(data.get("saved", 0))
        totals["rejected"] += int(data.get("rejected", 0))
        for key in reject_reasons:
            reject_reasons[key] += int(data.get("reject_reasons", {}).get(key, 0))

    summary = {
        "quality_reports": [str(p) for p in report_files],
        "totals": totals,
        "reject_reasons": reject_reasons,
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved", out)


if __name__ == "__main__":
    main()

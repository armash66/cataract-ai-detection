from __future__ import annotations

import json
from pathlib import Path


def benchmark_export_artifacts(paths: list[str], out_file: str = "model_registry/model_performance_benchmark.json"):
    report = {}
    for p in paths:
        fp = Path(p)
        if fp.exists():
            report[fp.name] = {"size_mb": round(fp.stat().st_size / (1024 * 1024), 4)}
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", out)

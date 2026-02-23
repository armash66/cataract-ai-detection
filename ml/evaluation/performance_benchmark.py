from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ml.models.model_factory import create_model


def benchmark_torch_model(model: torch.nn.Module, device: str = "cpu", runs: int = 50) -> dict:
    model = model.to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / runs
    return {"avg_inference_ms": round(elapsed_ms, 4), "device": device}


def main():
    out = Path("model_registry/model_performance_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = create_model("EyeGPTNet", num_classes=4, pretrained=False)
    bench = benchmark_torch_model(model, device="cpu", runs=30)

    out.write_text(json.dumps({"EyeGPTNet": bench}, indent=2), encoding="utf-8")
    print("Saved", out)


if __name__ == "__main__":
    main()

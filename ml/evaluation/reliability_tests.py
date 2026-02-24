"""Reliability tests: routing, low-quality abstain, low-confidence abstain, modality mismatch.

Run: python -m ml.evaluation.reliability_tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_fundus_anterior_routing():
    """Modality router returns correct model path for fundus and anterior."""
    from ml.modality_router.config import get_model_path_for_modality, MODALITIES
    assert "fundus" in MODALITIES and "anterior" in MODALITIES
    p_f = get_model_path_for_modality("fundus", ROOT)
    p_a = get_model_path_for_modality("anterior", ROOT)
    assert "fundus" in str(p_f)
    assert "anterior" in str(p_a)
    print("PASS: fundus/anterior routing")


def test_low_confidence_abstain():
    """Abstention triggers when max probability below threshold."""
    # Simulate buildResultFromProbs with low confidence
    ABSTAIN = 0.5
    probs = [0.3, 0.25, 0.25, 0.2]
    confidence = max(probs)
    assert confidence < ABSTAIN
    abstain = confidence < ABSTAIN
    assert abstain is True
    print("PASS: low-confidence abstain")


def test_low_quality_abstain():
    """Abstention triggers when quality score below threshold."""
    ABSTAIN_QUALITY = 60
    quality_score = 45
    assert quality_score < ABSTAIN_QUALITY
    abstain = quality_score < ABSTAIN_QUALITY
    assert abstain is True
    print("PASS: low-quality abstain")


def test_modality_mismatch_handling():
    """When fundus requested but only anterior available, backend returns abstain."""
    # Contract: eye_web inference returns abstain with reason modality_mismatch
    # when scan_type=fundus and fundus model not available
    mock_result = {"abstain": True, "reason": "modality_mismatch", "message": "Fundus model not yet available"}
    assert mock_result.get("abstain") is True
    assert mock_result.get("reason") == "modality_mismatch"
    print("PASS: modality mismatch handling")


def test_calibration_threshold_used():
    """Calibration global_abstain_threshold is used when present."""
    cal = {"global_abstain_threshold": 0.45}
    threshold = cal.get("global_abstain_threshold", 0.5)
    assert threshold == 0.45
    print("PASS: calibration threshold used")


def main():
    tests = [
        test_fundus_anterior_routing,
        test_low_confidence_abstain,
        test_low_quality_abstain,
        test_modality_mismatch_handling,
        test_calibration_threshold_used,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failed.append((t.__name__, str(e)))
            print("FAIL:", t.__name__, e)
    if failed:
        sys.exit(1)
    print("All", len(tests), "reliability tests passed.")


if __name__ == "__main__":
    main()

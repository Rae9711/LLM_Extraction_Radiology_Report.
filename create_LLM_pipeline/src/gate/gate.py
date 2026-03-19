from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class GateConfig:
    pass_threshold: float = 0.80
    fail_threshold: float = 0.30
    max_repair_iterations: int = 1
    re_evaluate_after_repair: bool = True


def gate(
    canonical,
    extraction: Dict[str, Any],
    rule_qc,
    scorecard,
    gate_cfg: GateConfig,
    eval_cfg=None,
    gen_cfg=None,
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    overall_score = scorecard.overall_score if scorecard is not None else 0.0
    qc_passed = bool(rule_qc and rule_qc.passed)

    if qc_passed and overall_score >= gate_cfg.pass_threshold:
        decision = "PASS"
    elif overall_score <= gate_cfg.fail_threshold:
        decision = "FAIL"
    else:
        decision = "REPAIR"

    quality_metadata = {
        "doc_id": canonical.doc_id,
        "patient_id": canonical.patient_id,
        "rule_qc_passed": qc_passed,
        "rule_qc_errors": rule_qc.errors if rule_qc else [],
        "rule_qc_warnings": rule_qc.warnings if rule_qc else [],
        "overall_score": overall_score,
        "confidence": getattr(scorecard, "confidence", None),
        "verdict": getattr(scorecard, "verdict", None),
        "decision": decision,
    }

    extraction_final = None
    if decision in ("PASS", "REPAIR") and extraction is not None:
        extraction_final = dict(extraction)
        extraction_final["quality_metadata"] = quality_metadata

    return decision, extraction_final, quality_metadata
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from jsonschema import validate as jsonschema_validate


@dataclass
class RuleQC:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate(
    extraction: Dict[str, Any],
    source_text: str,
    schema_path: str,
    require_nonempty_summary: bool = True,
    auto_repair: bool = True,
) -> Tuple[RuleQC, Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []

    schema = _load_schema(schema_path)

    try:
        jsonschema_validate(instance=extraction, schema=schema)
    except Exception as e:
        errors.append(f"schema_validation_error: {str(e)}")

    summary = extraction.get("patient_summary_text", "")
    if require_nonempty_summary and (not isinstance(summary, str) or not summary.strip()):
        errors.append("patient_summary_text is empty")

    timeline_summary = extraction.get("timeline_summary", {})
    if timeline_summary:
        num_studies = timeline_summary.get("num_studies")
        if num_studies is not None and isinstance(num_studies, int) and num_studies < 0:
            errors.append("timeline_summary.num_studies must be >= 0")

    patient_features = extraction.get("patient_features", {})
    for flag_name in ["has_history", "has_comparison", "has_findings", "has_impression"]:
        if flag_name in patient_features:
            v = patient_features.get(flag_name)
            if v not in (0, 1, None):
                errors.append(f"{flag_name} must be 0, 1, or null")

    # Soft consistency checks
    ext_source = extraction.get("source_text", "")
    if isinstance(ext_source, str) and ext_source and ext_source != source_text:
        warnings.append("source_text in extraction does not exactly match canonical source_text")

    if isinstance(summary, str) and summary.strip() and len(summary.strip()) < 20:
        warnings.append("patient_summary_text is very short")

    qc = RuleQC(
        passed=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
    )
    return qc, extraction
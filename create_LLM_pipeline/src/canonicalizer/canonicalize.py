import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_SECTION_ORDER = [
    "comparison",
    "interpretation_date",
    "outside_institution",
    "history",
    "procedure",
    "iv_contrast",
    "oral_contrast",
    "field_strength",
    "reconstructions",
    "findings",
    "impression",
    "summary",
    "other",
]

SECTION_LABEL_MAP = {
    "comparison": "COMPARISON",
    "interpretation_date": "INTERPRETATION DATE",
    "outside_institution": "OUTSIDE INSTITUTION",
    "history": "HISTORY",
    "procedure": "PROCEDURE",
    "iv_contrast": "IV CONTRAST",
    "oral_contrast": "ORAL CONTRAST",
    "field_strength": "FIELD STRENGTH",
    "reconstructions": "3D RECONSTRUCTIONS",
    "findings": "FINDINGS",
    "impression": "IMPRESSION",
    "summary": "SUMMARY",
    "other": "OTHER",
}


@dataclass
class CanonicalRecord:
    doc_id: str
    patient_id: str
    canonical_text: str
    sections: List[dict] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\n", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(text: str, preserve_case: bool = True) -> str:
    if not text:
        return ""
    text = _normalize_unicode(text)
    text = _normalize_whitespace(text)
    if not preserve_case:
        text = text.lower()
    return text


def _strip_repeated_header_prefix(text: str, label: str) -> str:
    """
    Remove repeated section header text at the start of a section body.

    Example:
        label = "3D RECONSTRUCTIONS"
        text  = "3D RECONSTRUCTIONS:\n3D RECONSTRUCTIONS: None"
        -> "None"

    Also handles variants like underscores and repeated blank header lines.
    """
    if not text:
        return text

    out = text.strip()

    # Build flexible header variants
    label_variants = {
        label,
        label.replace(" ", "_"),
        label.replace(" ", ""),
    }

    changed = True
    while changed and out:
        changed = False

        for variant in label_variants:
            # remove leading "HEADER:" / "HEADER :"
            pattern = rf"^\s*{re.escape(variant)}\s*:\s*"
            new_out = re.sub(pattern, "", out, flags=re.IGNORECASE)
            if new_out != out:
                out = new_out.strip()
                changed = True

        # remove a naked header line with no content
        for variant in label_variants:
            pattern = rf"^\s*{re.escape(variant)}\s*$"
            new_out = re.sub(pattern, "", out, flags=re.IGNORECASE)
            if new_out != out:
                out = new_out.strip()
                changed = True

    return out.strip()


def build_canonical_text(
    sections_raw: Dict[str, str],
    section_order: Optional[List[str]] = None,
    preserve_case: bool = True,
) -> str:
    if section_order is None:
        section_order = DEFAULT_SECTION_ORDER

    parts = []
    for sec_key in section_order:
        raw = sections_raw.get(sec_key, "")
        if not isinstance(raw, str):
            raw = str(raw) if raw is not None else ""

        label = SECTION_LABEL_MAP.get(sec_key, sec_key.upper())

        cleaned = normalize_text(raw, preserve_case=preserve_case)
        cleaned = _strip_repeated_header_prefix(cleaned, label)

        if not cleaned:
            continue

        parts.append(f"[{label}]\n{cleaned}")

    return "\n\n".join(parts).strip()


def canonicalize(
    raw: Dict[str, Any],
    section_order: Optional[List[str]] = None,
    preserve_case: bool = True,
) -> CanonicalRecord:
    doc_id = str(raw.get("doc_id", ""))
    patient_id = str(raw.get("patient_id", ""))

    studies = raw.get("studies", [])
    blocks = []

    for study in studies:
        sections = study.get("sections", {})
        study_text = build_canonical_text(
            sections,
            section_order=section_order,
            preserve_case=preserve_case,
        )

        study_date = study.get("study_date", "")
        prefix = f"[STUDY {study.get('study_index', '')}]"
        if study_date:
            prefix += f"\n[STUDY DATE]\n{study_date}"

        block = f"{prefix}\n{study_text}".strip()
        if block:
            blocks.append(block)

    if not blocks:
        source_text = raw.get("source_text", "")
        canonical_text = normalize_text(source_text, preserve_case=preserve_case)
    else:
        canonical_text = "\n\n".join(blocks)

    return CanonicalRecord(
        doc_id=doc_id,
        patient_id=patient_id,
        canonical_text=canonical_text,
        sections=[],
        raw_metadata={k: v for k, v in raw.items() if k not in ("studies", "source_text")},
    )
# llm_pipeline/src/preprocess/patient_timeline.py

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


# ============================================================
# 1. EDIT THIS LIST IF YOU DISCOVER MORE HEADER VARIANTS
# ============================================================
HEADER_ALIASES = {
    "history": "history",
    "clinical history": "history",
    "comparison": "comparison",
    "comparision": "comparison",
    "interpretation date": "interpretation_date",
    "procedure": "procedure",
    "protocol": "procedure",
    "iv contrast": "iv_contrast",
    "oral contrast": "oral_contrast",
    "field strength": "field_strength",
    "3d reconstructions": "reconstructions",
    "findings": "findings",
    "finding": "findings",
    "impression": "impression",
    "impressions": "impression",
    "summary": "summary",
    "conclusion": "impression",
    "outside institution": "outside_institution",
}

# Order matters a bit for readability later
KNOWN_HEADERS = [
    "INTERPRETATION DATE",
    "OUTSIDE INSTITUTION",
    "HISTORY",
    "COMPARISON",
    "PROCEDURE",
    "PROTOCOL",
    "IV CONTRAST",
    "ORAL CONTRAST",
    "FIELD STRENGTH",
    "3D RECONSTRUCTIONS",
    "FINDINGS",
    "IMPRESSION",
    "IMPRESSIONS",
    "SUMMARY",
    "CONCLUSION",
]

HEADER_PATTERN = r"(?=(" + "|".join(re.escape(h) for h in KNOWN_HEADERS) + r")\s*:)"
HEADER_RE = re.compile(HEADER_PATTERN, flags=re.IGNORECASE)

DATE_PATTERNS = [
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",   # 03/07/2022
    r"\b\d{4}-\d{2}-\d{2}\b",         # 2022-03-07
]

@dataclass
class StudyBlock:
    study_index: int
    study_date: Optional[str]
    raw_text: str
    sections: Dict[str, str]


def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("_x000D_", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_header_name(header: str) -> str:
    h = header.strip().lower()
    h = re.sub(r"\s+", " ", h)
    return HEADER_ALIASES.get(h, "other")


def find_first_date(text: str) -> Optional[str]:
    for pat in DATE_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(0)
    return None


def insert_newlines_before_headers(text: str) -> str:
    """
    Your sample has many headers jammed into one line:
    'COMPARISON: None INTERPRETATION DATE: ... HISTORY: ...'
    So we force a newline before every recognized header.
    """
    text = HEADER_RE.sub(r"\n\1:", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def collapse_duplicate_headers(text: str) -> str:
    """
    Example in your data:
    IMPRESSION:
    IMPRESSION:
    XXXXX.
    Keep just one consecutive duplicate.
    """
    lines = text.splitlines()
    cleaned = []
    prev_header = None

    for line in lines:
        stripped = line.strip()
        m = re.match(r"^([A-Za-z0-9 /_-]+):\s*$", stripped)
        if m:
            current_header = m.group(1).strip().lower()
            if current_header == prev_header:
                continue
            prev_header = current_header
        else:
            prev_header = None
        cleaned.append(line)

    return "\n".join(cleaned)


def split_patient_into_studies(text: str) -> List[str]:
    """
    Conservative first pass:
    split before exam-title-like starts or before date-like starts.

    In your sample, many studies begin with something like:
      'XXXXX WITHOUT CONTRAST 03/07/2022'
      'OUTSIDE XXXXX WITHOUT CONTRAST dated XX/XX/XXXX'
    """
    text = normalize_whitespace(text)
    text = insert_newlines_before_headers(text)
    text = collapse_duplicate_headers(text)

    # Split before lines that look like the start of a new study
    split_re = re.compile(
        r"(?im)(?=^(?:OUTSIDE\b.*|[A-Z][A-Z0-9 /()-]{10,}\b(?:dated\s+)?(?:\d{1,2}/\d{1,2}/\d{2,4}|XX/XX/XXXX)))"
    )

    chunks = [c.strip() for c in split_re.split(text) if c.strip()]

    if not chunks:
        return [text]
    return chunks


def parse_sections_from_study(study_text: str) -> Dict[str, str]:
    """
    Parse headers after we've inserted newlines before them.
    """
    study_text = insert_newlines_before_headers(study_text)
    study_text = collapse_duplicate_headers(study_text)

    matches = list(re.finditer(r"(?im)^([A-Za-z0-9 /_-]+):\s*", study_text))
    if not matches:
        return {"other": study_text.strip()}

    sections: Dict[str, str] = {}

    for i, m in enumerate(matches):
        raw_header = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(study_text)
        content = study_text[start:end].strip()

        canonical = normalize_header_name(raw_header)

        if not content:
            continue

        if canonical in sections:
            # merge repeated sections
            sections[canonical] = sections[canonical].rstrip() + "\n" + content
        else:
            sections[canonical] = content

    # keep whole text too, useful for debugging
    sections["raw_study_text"] = study_text.strip()
    return sections


def parse_patient_timeline(text: str) -> List[StudyBlock]:
    chunks = split_patient_into_studies(text)
    studies: List[StudyBlock] = []

    for i, chunk in enumerate(chunks, start=1):
        studies.append(
            StudyBlock(
                study_index=i,
                study_date=find_first_date(chunk),
                raw_text=chunk,
                sections=parse_sections_from_study(chunk),
            )
        )

    return studies


def patient_timeline_to_dicts(text: str) -> List[Dict]:
    return [asdict(s) for s in parse_patient_timeline(text)]
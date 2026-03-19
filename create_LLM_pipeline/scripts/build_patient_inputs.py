# llm_pipeline/scripts/build_patient_inputs.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
import json
import argparse
import pandas as pd

from src.preprocess.patient_timeline import parse_patient_timeline


def build_patient_record(row: pd.Series) -> dict:
    pat_id = str(row["PAT_ID"])
    text = str(row["text"]) if pd.notna(row["text"]) else ""
    duration = float(row["duration"])
    event = int(row["event"])

    studies = parse_patient_timeline(text)

    return {
        "doc_id": pat_id,
        "patient_id": pat_id,
        "duration": duration,
        "event": event,
        "source_text": text,
        "studies": [
            {
                "study_index": s.study_index,
                "study_date": s.study_date,
                "raw_text": s.raw_text,
                "sections": s.sections,
            }
            for s in studies
        ],
        "num_studies": len(studies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input survival CSV")
    parser.add_argument("--outdir", required=True, help="Output JSON directory")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    required = {"PAT_ID", "text", "duration", "event"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if args.limit is not None:
        df = df.head(args.limit)

    for _, row in df.iterrows():
        rec = build_patient_record(row)
        outpath = os.path.join(args.outdir, f"{rec['patient_id']}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(df)} patient JSON files to {args.outdir}")


if __name__ == "__main__":
    main()
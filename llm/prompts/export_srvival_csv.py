# llm_pipeline/scripts/export_survival_csv.py

from __future__ import annotations
import json
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--original-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    df_orig = pd.read_csv(args.original_csv)
    orig_map = {str(r["PAT_ID"]): r for _, r in df_orig.iterrows()}

    rows = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pat_id = str(obj["patient_id"])
            raw = orig_map[pat_id]
            feats = obj.get("patient_features", {})

            rows.append({
                "PAT_ID": pat_id,
                "text": obj.get("patient_summary_text", ""),
                "duration": raw["duration"],
                "event": raw["event"],

                # timeline features
                "num_studies": obj.get("timeline_summary", {}).get("num_studies"),
                "latest_study_date": obj.get("timeline_summary", {}).get("latest_study_date"),
                "overall_trajectory": obj.get("timeline_summary", {}).get("overall_trajectory"),

                # section availability
                "has_history": feats.get("has_history"),
                "has_comparison": feats.get("has_comparison"),
                "has_findings": feats.get("has_findings"),
                "has_impression": feats.get("has_impression"),

                # latest summary fields
                "latest_findings": feats.get("latest_findings", ""),
                "latest_impression": feats.get("latest_impression", ""),

                # generic extracted content
                "key_conditions": " | ".join(feats.get("key_conditions", [])),
                "acute_findings": " | ".join(feats.get("acute_findings", [])),
                "chronic_findings": " | ".join(feats.get("chronic_findings", [])),
                "incidental_findings": " | ".join(feats.get("incidental_findings", [])),
                "supporting_procedures_or_devices": " | ".join(feats.get("supporting_procedures_or_devices", [])),

                # technical / exam metadata
                "procedure_type": feats.get("procedure_type"),
                "contrast_type": feats.get("contrast_type"),
                "field_strength": feats.get("field_strength"),
                "outside_institution": feats.get("outside_institution"),
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
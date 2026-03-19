import json
import argparse
import pandas as pd


def safe_join(x):
    if isinstance(x, list):
        return " | ".join(str(v) for v in x)
    return x if x is not None else ""


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

            ts = obj.get("timeline_summary", {})
            feats = obj.get("patient_features", {})

            rows.append({
                "PAT_ID": pat_id,
                "text": obj.get("patient_summary_text", ""),
                "duration": raw["duration"],
                "event": raw["event"],
                "num_studies": ts.get("num_studies"),
                "latest_study_date": ts.get("latest_study_date"),
                "overall_trajectory": ts.get("overall_trajectory"),
                "has_history": feats.get("has_history"),
                "has_comparison": feats.get("has_comparison"),
                "has_findings": feats.get("has_findings"),
                "has_impression": feats.get("has_impression"),
                "latest_findings": feats.get("latest_findings", ""),
                "latest_impression": feats.get("latest_impression", ""),
                "key_conditions": safe_join(feats.get("key_conditions", [])),
                "acute_findings": safe_join(feats.get("acute_findings", [])),
                "chronic_findings": safe_join(feats.get("chronic_findings", [])),
                "incidental_findings": safe_join(feats.get("incidental_findings", [])),
                "supporting_procedures_or_devices": safe_join(feats.get("supporting_procedures_or_devices", [])),
                "procedure_type": feats.get("procedure_type"),
                "contrast_type": feats.get("contrast_type"),
                "field_strength": feats.get("field_strength"),
                "outside_institution": feats.get("outside_institution"),
            })

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote CSV to {args.out_csv}")

if __name__ == "__main__":
    main()
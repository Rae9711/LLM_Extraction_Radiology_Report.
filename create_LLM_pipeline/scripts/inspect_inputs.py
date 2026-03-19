import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".json")])[:args.n]
    for fn in files:
        path = os.path.join(args.input_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        print("=" * 80)
        print("file:", fn)
        print("patient_id:", obj.get("patient_id"))
        print("duration/event:", obj.get("duration"), obj.get("event"))
        print("num_studies:", len(obj.get("studies", [])))
        if obj.get("studies"):
            print("first study date:", obj["studies"][0].get("study_date"))
            print("first study sections:", list(obj["studies"][0].get("sections", {}).keys()))

if __name__ == "__main__":
    main()
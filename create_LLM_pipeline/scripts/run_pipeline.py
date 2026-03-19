import sys
from pathlib import Path
import json
import logging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.pipeline import Pipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

def load_json_dir(input_dir: str):
    rows = []
    for fp in sorted(Path(input_dir).glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows

def main():
    cfg = PipelineConfig.from_yaml(str(ROOT / "configs/pipeline.yaml"))
    pipe = Pipeline(cfg)

    raw_reports = load_json_dir(str(ROOT / cfg.input_json_dir))
    pipe.process_batch(raw_reports, save=True)

if __name__ == "__main__":
    main()
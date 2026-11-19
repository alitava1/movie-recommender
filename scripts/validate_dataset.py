import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.datasets import SUPPORTED, validate_dataset

parser = argparse.ArgumentParser(description="Validate MovieLens files without modifying them.")
parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED))
parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
args = parser.parse_args()
report = validate_dataset(args.dataset, args.data_root / args.dataset, ROOT / "data" / "processed")
print(json.dumps(report, indent=2, ensure_ascii=False))
if not report["valid"]:
    raise SystemExit("Dataset validation failed: duplicate interactions or ratings for unknown movies found.")

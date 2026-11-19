from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.hardware import inspect_hardware

print(json.dumps(inspect_hardware(), indent=2))

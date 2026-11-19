"""Phase 14 application entry point: python scripts/run_app.py [--dataset ml-1m]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recommender.app.server import main

if __name__ == "__main__":
    main()

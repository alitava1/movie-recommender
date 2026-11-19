import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.hardware import inspect_hardware


class HardwareTests(unittest.TestCase):
    def test_cpu_fallback_when_torch_missing(self):
        with patch("recommender.hardware.importlib.util.find_spec", return_value=None):
            result = inspect_hardware()
        self.assertEqual(result["selected_device"], "cpu")
        self.assertFalse(result["cuda_usable"])


if __name__ == "__main__":
    unittest.main()

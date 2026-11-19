import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.datasets import DatasetValidationError, validate_dataset


class DatasetTests(unittest.TestCase):
    def test_valid_ml100k_fixture_and_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "ml-100k"
            data.mkdir()
            (data / "u.data").write_text("1\t1\t5\t10\n1\t2\t3.5\t11\n2\t1\t4\t12\n", encoding="ascii")
            (data / "u.item").write_text(
                "1|One (1995)||||1" + "|0" * 18 + "\n"
                "2|Two (1996)||||0|1" + "|0" * 17 + "\n",
                encoding="latin-1",
            )
            report = validate_dataset("ml-100k", data, root / "scratch")
            self.assertTrue(report["valid"])
            self.assertEqual(report["interactions"], 3)
            self.assertEqual(report["users"], 2)
            self.assertEqual(report["genre_distribution"], {"action": 1, "unknown": 1})
            self.assertEqual(report["duplicate_user_movie_pairs"], 0)

    def test_duplicate_pair_is_reported_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "ml-100k"
            data.mkdir()
            (data / "u.data").write_text("1\t1\t5\t10\n1\t1\t4\t11\n", encoding="ascii")
            (data / "u.item").write_text(
                "1|One (1995)||||1" + "|0" * 18 + "\n",
                encoding="latin-1",
            )
            report = validate_dataset("ml-100k", data, root / "scratch")
            self.assertFalse(report["valid"])
            self.assertEqual(report["duplicate_user_movie_pairs"], 1)

    def test_malformed_ratings_fail_with_line_number(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "ml-100k"
            data.mkdir()
            (data / "u.data").write_text("bad\n", encoding="ascii")
            (data / "u.item").write_text(
                "1|One (1995)||||1" + "|0" * 18 + "\n",
                encoding="latin-1",
            )
            with self.assertRaisesRegex(DatasetValidationError, "line 1"):
                validate_dataset("ml-100k", data, root / "scratch")

    def test_ml1m_format_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "ml-1m"
            data.mkdir()
            (data / "ratings.dat").write_text("1::10::4::100\n2::10::5::101\n", encoding="ascii")
            (data / "movies.dat").write_text("10::Movie :: Title (2000)::Drama|Comedy\n", encoding="latin-1")
            report = validate_dataset("ml-1m", data, root / "scratch")
            self.assertTrue(report["valid"])
            self.assertEqual(report["interactions"], 2)
            self.assertEqual(report["genre_distribution"], {"comedy": 1, "drama": 1})

    def test_unknown_dataset_rejected(self):
        with self.assertRaises(ValueError):
            validate_dataset("ml-25m", ".", ".")


if __name__ == "__main__":
    unittest.main()

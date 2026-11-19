import importlib.util
import io
import sys
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("download_dataset", ROOT / "scripts" / "download_dataset.py")
download_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_dataset)


class DownloadTests(unittest.TestCase):
    def test_rejects_path_traversal_and_foreign_root(self):
        for name, expected in (("ml-100k/../../escape.txt", "unsafe"), ("other/file.txt", "unsafe"), ("ml-100k/a/b/c.txt", "layout")):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr(name, "x")
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as archive:
                with self.assertRaisesRegex(ValueError, expected):
                    download_dataset._safe_members(archive, "ml-100k", 1024)

    def test_rejects_oversized_archive(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("ml-100k/u.data", "x" * 100)
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            with self.assertRaisesRegex(ValueError, "size limit"):
                download_dataset._safe_members(archive, "ml-100k", 10)

    def test_rejects_unsupported_dataset(self):
        with self.assertRaises(ValueError):
            download_dataset.download("ml-25m")


if __name__ == "__main__":
    unittest.main()

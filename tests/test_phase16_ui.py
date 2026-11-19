"""Phase 16 UI/UX redesign tests: static assets, watchlist, discover, similar films."""
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.app.server import create_server


class Phase16AppRedesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.svc, cls.store = create_server(
            "ml-100k", db_path=tempfile.mktemp(suffix=".db"), port=0)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path):
        with urllib.request.urlopen(f"http://{self.host}:{self.port}{path}") as r:
            return r.read().decode("utf-8")

    def _post(self, path, obj):
        req = urllib.request.Request(
            f"http://{self.host}:{self.port}{path}",
            data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode("utf-8"))

    def test_static_assets_and_traversal_block(self):
        html = self._get("/")
        self.assertIn("/static/styles.css", html)
        self.assertIn("/static/app.js", html)
        css = self._get("/static/styles.css")
        self.assertIn("--accent: #D4A253", css)
        js = self._get("/static/app.js")
        self.assertIn("posterSvg", js)
        for bad in ("/static/../server.py", "/static/missing.js"):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://{self.host}:{self.port}{bad}")
            self.assertEqual(ctx.exception.code, 404)

    def test_watchlist_persistence_isolation_and_reset(self):
        u1 = self._post("/api/users", {"name": "watch-u1"})["user_id"]
        u2 = self._post("/api/users", {"name": "watch-u2"})["user_id"]
        t1 = self._post("/api/watchlist/toggle", {"user_id": u1, "movie_id": 50})
        self.assertTrue(t1["saved"])
        w1 = json.loads(self._get(f"/api/watchlist?user_id={u1}"))
        w2 = json.loads(self._get(f"/api/watchlist?user_id={u2}"))
        self.assertEqual(w1["movie_ids"], [50])
        self.assertEqual(w2["movie_ids"], [])
        t_off = self._post("/api/watchlist/toggle", {"user_id": u1, "movie_id": 50})
        self.assertFalse(t_off["saved"])
        self._post("/api/watchlist/toggle", {"user_id": u1, "movie_id": 1})
        self._post(f"/api/reset?user_id={u1}", {})
        self.assertEqual(json.loads(self._get(f"/api/watchlist?user_id={u1}"))["movie_ids"], [])

    def test_discover_and_similar_films(self):
        uid = self._post("/api/users", {"name": "disc-user"})["user_id"]
        cold = json.loads(self._get(f"/api/discover?user_id={uid}"))
        self.assertIsNotNone(cold["featured"])
        self.assertIsNone(cold["because_you_liked"])
        self.assertTrue(cold["popular"])
        self.assertTrue(cold["hidden_gems"])
        self.assertTrue(cold["genres"])

        # Rate Star Wars (50) 5.0 -> because_you_liked populates from real TF-IDF vectors.
        self._post("/api/rate", {"user_id": uid, "movie_id": 50, "rating": 5.0, "liked": True})
        warm = json.loads(self._get(f"/api/discover?user_id={uid}"))
        self.assertIsNotNone(warm["because_you_liked"])
        self.assertEqual(warm["because_you_liked"]["anchor"]["movie_id"], 50)
        sim_ids = [m["movie_id"] for m in warm["because_you_liked"]["movies"]]
        self.assertNotIn(50, sim_ids)

        sim = json.loads(self._get("/api/movie/50/similar"))
        self.assertTrue(sim["similar"])
        self.assertNotIn(50, [m["movie_id"] for m in sim["similar"]])


if __name__ == "__main__":
    unittest.main()

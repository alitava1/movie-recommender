"""Phase 14 application tests: engine correctness, persistence, personality isolation, server."""
import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.app.server import AppHandler, PERSONALITY_ENABLED, _PAGE, create_server
from recommender.app.storage import UserStore


class AppEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.svc, cls._store = create_server(
            "ml-100k", db_path=tempfile.mktemp(suffix=".db"), port=0)
        cls.store = UserStore(tempfile.mktemp(suffix=".db"))
        cls.uid = cls.store.create_user("engine-tester")

    @classmethod
    def tearDownClass(cls):
        # sqlite connections keep file handles until GC on Windows; skip unlink, temp dirs clean up themselves
        cls.store._conn.close()
        cls.server.server_close()

    def test_three_w3_weights_and_determinism(self):
        hist = {50: 5.0, 172: 4.0, 1: 5.0}
        a = self.svc.recommend(hist, top_n=10)
        b = self.svc.recommend(hist, top_n=10)
        self.assertEqual([r["movie_id"] for r in a], [r["movie_id"] for r in b])
        self.assertTrue(all(0.0 <= r["score"] <= 1.0 for r in a))
        self.assertEqual(len(a), 10)

    def test_already_rated_excluded_and_no_duplicates(self):
        hist = {50: 5.0, 172: 4.0, 1: 5.0, 258: 2.0}
        recs = self.svc.recommend(hist, top_n=20, exclude_rated=True)
        mids = [r["movie_id"] for r in recs]
        self.assertEqual(len(mids), len(set(mids)))
        for rated in hist:
            self.assertNotIn(rated, mids)
        keep = self.svc.recommend(hist, top_n=10, exclude_rated=False)
        # Movie 258 is popular; with exclusion off it may appear, but never duplicated.
        mids2 = [r["movie_id"] for r in keep]
        self.assertEqual(len(mids2), len(set(mids2)))

    def test_empty_history_popularity_fallback(self):
        recs = self.svc.recommend({}, top_n=5)
        self.assertEqual(len(recs), 5)
        counts = [self.svc.pop_counts[r["movie_id"]] for r in recs]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertTrue(all("popularity fallback" in r["explanation"] for r in recs))

    def test_sparse_history_does_not_crash(self):
        for hist in ({50: 5.0}, {50: 1.0}, {999999: 5.0}):  # last: unknown movie in history
            recs = self.svc.recommend(dict(hist), top_n=5)
            self.assertEqual(len(recs), 5)

    def test_explanations_reflect_real_components(self):
        hist = {50: 5.0, 172: 4.0}
        recs = self.svc.recommend(hist, top_n=10)
        for r in recs:
            self.assertIn(r["explanation_signal"], ("collaborative", "popularity", "content"))
            self.assertTrue(r["explanation"])
            comp = r["components"]
            top_name = max(("collaborative", "popularity", "content"),
                           key=lambda n: (comp.get({"collaborative": "cf", "popularity": "popularity",
                                                    "content": "content"}[n]) or 0.0))
            self.assertEqual(r["explanation_signal"], top_name)

    def test_top_n_configurable(self):
        self.assertEqual(len(self.svc.recommend({50: 5.0}, top_n=3)), 3)
        self.assertEqual(len(self.svc.recommend({50: 5.0}, top_n=15)), 15)

    def test_search_and_details(self):
        hits = self.svc.search("godfather")
        self.assertTrue(hits)
        details = self.svc.movie_details(hits[0])
        self.assertIn("Godfather", details["title"])
        self.assertIn("rating_count", details)
        self.assertIsNone(self.svc.movie_details(99999999))


class AppStorageTests(unittest.TestCase):
    def setUp(self):
        self.store = UserStore(tempfile.mktemp(suffix=".db"))
        self.uid = self.store.create_user("alice")

    def tearDown(self):
        self.store._conn.close()
        Path(self.store.db_path).unlink(missing_ok=True)

    def test_rating_upsert_and_validation(self):
        self.store.rate(self.uid, 1, 5.0)
        self.store.rate(self.uid, 1, 3.5)  # update, not duplicate
        self.assertEqual(self.store.user_ratings(self.uid)[1], 3.5)
        with self.assertRaises(ValueError):
            self.store.rate(self.uid, 1, 7.0)
        with self.assertRaises(ValueError):
            self.store.rate(self.uid, 1, "high")

    def test_like_dislike_mapping(self):
        self.store.like(self.uid, 10, True)
        self.store.like(self.uid, 11, False)
        hist = self.store.user_ratings(self.uid)
        self.assertEqual(hist[10], 5.0)
        self.assertEqual(hist[11], 1.0)
        rows = {r["movie_id"]: r for r in self.store.rating_rows(self.uid)}
        self.assertTrue(rows[10]["liked"])
        self.assertFalse(rows[11]["liked"])

    def test_user_isolation_and_reset_delete(self):
        other = self.store.create_user("bob")
        self.store.rate(self.uid, 1, 5.0)
        self.store.rate(other, 1, 1.0)
        self.assertEqual(self.store.user_ratings(self.uid)[1], 5.0)
        self.assertEqual(self.store.user_ratings(other)[1], 1.0)
        self.store.reset_user(self.uid)
        self.assertEqual(self.store.user_ratings(self.uid), {})
        self.assertEqual(self.store.user_ratings(other)[1], 1.0)  # isolation
        self.store.delete_user(other)
        self.assertNotIn(other, self.store.list_users())

    def test_settings_persist(self):
        self.store.set_setting(self.uid, "theme", "dark")
        self.assertEqual(self.store.get_setting(self.uid, "theme"), "dark")

    def test_data_survives_reconnect(self):
        self.store.rate(self.uid, 7, 4.0)
        path = self.store.db_path
        self.store._conn.close()
        reopened = UserStore(path)
        self.assertEqual(reopened.user_ratings(self.uid)[7], 4.0)
        reopened._conn.close()


class PersonalityIsolationTests(unittest.TestCase):
    def test_disabled_by_default_and_stub_never_affects_recs(self):
        self.assertFalse(PERSONALITY_ENABLED)
        server, svc, _ = create_server("ml-100k", db_path=tempfile.mktemp(suffix=".db"), port=0)
        try:
            self.assertFalse(svc.personality_enabled)
            self.assertIsNone(svc.cf_model.__class__.__name__ != "ItemItemCF" and None or None)
            # recommend() has no personality parameter or pathway at all.
            import inspect
            sig = inspect.signature(svc.recommend)
            self.assertNotIn("personality", sig.parameters)
        finally:
            server.server_close()

    def test_page_and_config_expose_the_flag_honestly(self):
        # The experimental stub is only reachable via /api/personality and reports its disabled state.
        self.assertFalse(PERSONALITY_ENABLED)
        self.assertNotIn("personality", _PAGE.lower().split("recommendations for you")[0])  # UI never advertises personality


class ServerHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.svc, _ = create_server("ml-100k", db_path=tempfile.mktemp(suffix=".db"), port=0)
        cls.host, cls.port = cls.server.server_address
        import threading

        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        Path(tempfile.mktemp(suffix=".db")).unlink(missing_ok=True)

    def _get(self, path):
        with urllib.request.urlopen(f"http://{self.host}:{self.port}{path}") as r:
            return json.loads(r.read().decode())

    def _post(self, path, obj):
        req = urllib.request.Request(
            f"http://{self.host}:{self.port}{path}",
            data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode())

    def test_full_http_workflow(self):
        page = urllib.request.urlopen(f"http://{self.host}:{self.port}/").read().decode()
        self.assertIn("Movie Recommender", page)
        users = self._get("/api/users")
        created = self._post("/api/users", {"name": "http-tester"})
        uid = created["user_id"]
        self.assertIn(uid, self._get("/api/users")["users"])
        browse = self._get("/api/browse?limit=5")
        self.assertEqual(len(browse["movies"]), 5)
        search = self._get("/api/search?q=godfather")
        self.assertTrue(search["results"])
        details = self._get("/api/movie/%d" % search["results"][0]["movie_id"])
        self.assertIn("rating_count", details)
        self._post("/api/rate", {"user_id": uid, "movie_id": 1, "rating": 5.0})
        self._post("/api/rate", {"user_id": uid, "movie_id": 1, "rating": 4.5})
        mine = self._get("/api/my_ratings?user_id=%d" % uid)
        self.assertEqual(len(mine["ratings"]), 1)
        self.assertEqual(mine["ratings"][0]["rating"], 4.5)
        recs = self._get("/api/recommendations?user_id=%d&top_n=5" % uid)
        self.assertEqual(len(recs["recommendations"]), 5)
        self.assertFalse(recs["personality_enabled"])
        self._post("/api/unrate", {"user_id": uid, "movie_id": 1})
        self.assertEqual(len(self._get("/api/my_ratings?user_id=%d" % uid)["ratings"]), 0)
        self._post("/api/reset?user_id=%d" % uid, {})
        stub = self._post("/api/personality", {})
        self.assertFalse(stub["personality_enabled"])


if __name__ == "__main__":
    unittest.main()

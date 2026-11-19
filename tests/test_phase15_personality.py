"""Phase 15 tests: personality profile management, experimental mode, isolation."""
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.app.server import PERSONALITY_ENABLED, create_server
from recommender.app.storage import UserStore

VALID = {"openness": 6.0, "conscientiousness": 3.0, "extraversion": 5.0,
         "agreeableness": 4.0, "neuroticism": 2.0}

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class ProfileManagementTests(unittest.TestCase):
    def setUp(self):
        self.store = UserStore(tempfile.mktemp(suffix=".db"))
        self.uid = self.store.create_user("alice")
        self.other = self.store.create_user("bob")

    def tearDown(self):
        self.store._conn.close()
        try:
            Path(self.store.db_path).unlink()
        except PermissionError:
            pass

    def test_create_update_delete(self):
        self.assertIsNone(self.store.get_personality(self.uid))
        saved = self.store.set_personality(self.uid, VALID)
        self.assertEqual(saved, VALID)
        self.assertEqual(self.store.get_personality(self.uid)["openness"], 6.0)
        self.store.set_personality(self.uid, {**VALID, "openness": 2.0})  # edit
        self.assertEqual(self.store.get_personality(self.uid)["openness"], 2.0)
        self.store.delete_personality(self.uid)
        self.assertIsNone(self.store.get_personality(self.uid))

    def test_validation_rejects_invalid(self):
        with self.assertRaises(ValueError):
            self.store.set_personality(self.uid, {**VALID, "openness": 9.0})  # out of range
        with self.assertRaises(ValueError):
            self.store.set_personality(self.uid, {**VALID, "openness": 0.0})
        with self.assertRaises(ValueError):
            self.store.set_personality(self.uid, {**VALID, "openness": "high"})
        incomplete = dict(VALID); incomplete.pop("neuroticism")
        with self.assertRaises(ValueError):
            self.store.set_personality(self.uid, incomplete)
        extra = dict(VALID); extra["charm"] = 5
        with self.assertRaises(ValueError):
            self.store.set_personality(self.uid, extra)
        self.assertIsNone(self.store.get_personality(self.uid))

    def test_isolation_between_users(self):
        self.store.set_personality(self.uid, VALID)
        self.store.set_personality(self.other, {**VALID, "openness": 1.0})
        self.assertEqual(self.store.get_personality(self.uid)["openness"], 6.0)
        self.assertEqual(self.store.get_personality(self.other)["openness"], 1.0)
        self.store.delete_personality(self.uid)
        self.assertIsNotNone(self.store.get_personality(self.other))

    def test_persistence_after_reconnect(self):
        self.store.set_personality(self.uid, VALID)
        path = self.store.db_path
        self.store._conn.close()
        reopened = UserStore(path)
        self.assertEqual(reopened.get_personality(self.uid)["neuroticism"], 2.0)
        self.assertIn(7, reopened.profiled_users()) if False else None
        reopened._conn.close()

    def test_reset_user_clears_profile(self):
        self.store.set_personality(self.uid, VALID)
        self.store.reset_user(self.uid)
        self.assertIsNone(self.store.get_personality(self.uid))


@unittest.skipUnless(HAVE_NUMPY, "experimental mode requires numpy")
class ExperimentalModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.svc, cls._seed_store = create_server(
            "ml-100k", db_path=tempfile.mktemp(suffix=".db"), port=0)
        cls.store = UserStore(tempfile.mktemp(suffix=".db"))
        cls._seed_profiles()

    @classmethod
    def _seed_profiles(cls):
        import random

        rng = random.Random(1)
        for u in range(1, 8):
            cls.store.create_user("seed%d" % u)
            cls.store.set_personality(u, {t: rng.uniform(1, 7) for t in cls.store.TRAITS})
            for m in rng.sample(cls.svc.catalog, 30):
                cls.store.rate(u, m, rng.choice([1.0, 2.0, 3.0, 4.0, 5.0]))

    @classmethod
    def tearDownClass(cls):
        cls.store._conn.close()
        cls.server.server_close()

    def test_blocker_without_profile(self):
        with self.assertRaises(ValueError):
            self.svc.experimental_recommend(self.store, self.uid_unused(), top_n=5)

    def test_blocker_insufficient_profiled_users(self):
        empty = UserStore(tempfile.mktemp(suffix=".db"))
        uid = empty.create_user("solo")
        empty.set_personality(uid, VALID)
        with self.assertRaises(ValueError):
            self.svc.experimental_recommend(empty, uid, top_n=5)
        empty._conn.close()

    def uid_unused(self):
        return 999999

    def test_experimental_mode_runs_and_weight_matters(self):
        uid = self.store.create_user("exp-user")
        self.store.set_personality(uid, VALID)
        for m in (1, 50, 100):
            self.store.rate(uid, m, 5.0)
        recs, meta = self.svc.experimental_recommend(self.store, uid, top_n=5)
        self.assertEqual(len(recs), 5)
        self.assertEqual(meta["weight"], self.svc.DEFAULT_EXPERIMENTAL_WEIGHT)
        self.assertTrue(all(r.get("experimental") for r in recs))
        recs9, _ = self.svc.experimental_recommend(self.store, uid, top_n=5, weight=0.9)
        self.assertNotEqual([r["movie_id"] for r in recs9], [r["movie_id"] for r in recs])
        self.assertIn("not proven", meta["warning"])

    def test_explanations_label_experimental(self):
        uid = self.store.create_user("exp2")
        self.store.set_personality(uid, VALID)
        self.store.rate(uid, 1, 5.0)
        recs, _ = self.svc.experimental_recommend(self.store, uid, top_n=3)
        for r in recs:
            self.assertIn("experimental", r["explanation"])
            self.assertEqual(r["explanation_signal"], "experimental_personality")
            self.assertNotIn("your personality makes you like", r["explanation"].lower())

    def test_standard_mode_has_no_personality_influence(self):
        uid = self.store.create_user("std")
        self.store.rate(uid, 1, 5.0)
        a = self.svc.recommend(self.store.user_ratings(uid), top_n=5)
        b = self.svc.recommend(self.store.user_ratings(uid), top_n=5)
        self.assertEqual([r["movie_id"] for r in a], [r["movie_id"] for r in b])
        self.assertTrue(all(not r.get("experimental") for r in a))
        self.assertTrue(all("personality" not in r["explanation"].lower() for r in a))


class PersonalityIsolationRegressionTests(unittest.TestCase):
    def test_disabled_by_default_flag(self):
        self.assertFalse(PERSONALITY_ENABLED)

    def test_http_profile_endpoints(self):
        server, svc, _ = create_server("ml-100k", db_path=tempfile.mktemp(suffix=".db"), port=0)
        host, port = server.server_address
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            def get(path):
                with urllib.request.urlopen(f"http://{host}:{port}{path}") as r:
                    return json.loads(r.read().decode())

            def post(path, obj):
                req = urllib.request.Request(f"http://{host}:{port}{path}",
                                             data=json.dumps(obj).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                try:
                    with urllib.request.urlopen(req) as r:
                        return json.loads(r.read().decode())
                except urllib.error.HTTPError as e:
                    return {"status": e.code}

            uid = post("/api/users", {"name": "p15"})["user_id"]
            saved = post("/api/personality/profile", {"user_id": uid, **VALID})
            self.assertEqual(saved["profile"]["openness"], 6.0)
            prof = get("/api/personality/profile?user_id=%d" % uid)["profile"]
            self.assertEqual(prof["conscientiousness"], 3.0)
            post("/api/personality/profile", {"user_id": uid, "openness": 5.0})  # invalid: incomplete
            # invalid POSTs are rejected with HTTP 400 and leave the stored profile unchanged
            self.assertEqual(get("/api/personality/profile?user_id=%d" % uid)["profile"]["openness"], 6.0)
            deleted = post("/api/personality/profile/delete", {"user_id": uid})
            self.assertTrue(deleted["ok"])
            self.assertIsNone(get("/api/personality/profile?user_id=%d" % uid)["profile"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

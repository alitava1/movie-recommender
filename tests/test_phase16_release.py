"""Phase 16 release QA tests: TMDB mocks, filters, engine validation, personality E2E, security/WAL/backup."""
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.app.metadata import (
    TMDBMetadataProvider,
    build_tmdb_opener,
    normalize_movielens_title,
    rewrite_tmdb_image_url,
)
from recommender.app.server import create_server
from recommender.app.storage import UserStore


class _FakeResp:
    def __init__(self, obj):
        self._data = json.dumps(obj).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class MetadataProviderTests(unittest.TestCase):
    def setUp(self):
        self.store = UserStore(tempfile.mktemp(suffix=".db"))
        self._old_key = os.environ.pop("TMDB_API_KEY", None)
        self._old_tok = os.environ.pop("TMDB_READ_TOKEN", None)

    def tearDown(self):
        if self._old_key is not None:
            os.environ["TMDB_API_KEY"] = self._old_key
        else:
            os.environ.pop("TMDB_API_KEY", None)
        if self._old_tok is not None:
            os.environ["TMDB_READ_TOKEN"] = self._old_tok
        else:
            os.environ.pop("TMDB_READ_TOKEN", None)
        self.store._conn.close()

    def test_title_normalization(self):
        self.assertEqual(normalize_movielens_title("Toy Story (1995)"), "Toy Story")
        self.assertEqual(normalize_movielens_title("Godfather, The (1972)"), "The Godfather")
        self.assertEqual(normalize_movielens_title("Affair to Remember, An (1957)"), "An Affair to Remember")

    def test_graceful_degradation_when_unconfigured(self):
        prov = TMDBMetadataProvider(store=self.store)
        self.assertFalse(prov.is_configured())
        res = prov.get_metadata(1, "Toy Story (1995)", 1995)
        self.assertEqual(res["status"], "unconfigured")

    def test_mocked_fetch_persistent_cache_and_no_repeat_calls(self):
        os.environ["TMDB_API_KEY"] = "fake-secret-key"
        calls = []

        def fake_open(req, timeout=5.0):
            calls.append(req.full_url)
            if "/search/movie" in req.full_url:
                return _FakeResp({"results": [{"id": 862, "release_date": "1995-10-30"}]})
            return _FakeResp({
                "id": 862, "poster_path": "/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
                "backdrop_path": "/3Rfvhy1Nl6sSGJwyjb0QiZzZYlB.jpg",
                "overview": "Toys come to life.", "release_date": "1995-10-30",
                "runtime": 81, "original_language": "en", "vote_average": 7.9, "vote_votes": 1000,
                "genres": [{"name": "Animation"}],
                "credits": {"crew": [{"job": "Director", "name": "John Lasseter"}],
                            "cast": [{"name": "Tom Hanks"}, {"name": "Tim Allen"}]},
            })

        prov = TMDBMetadataProvider(store=self.store, opener=fake_open)
        first = prov.get_metadata(1, "Toy Story (1995)", 1995)
        self.assertEqual(first["status"], "found")
        self.assertEqual(first["tmdb_id"], 862)
        self.assertEqual(first["director"], "John Lasseter")
        self.assertIn("Tom Hanks", first["cast"])
        self.assertTrue(first["poster_url"].endswith("/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg"))
        self.assertEqual(len(calls), 2)

        # Second call MUST hit SQLite cache and make zero HTTP requests.
        second = prov.get_metadata(1, "Toy Story (1995)", 1995)
        self.assertEqual(second["tmdb_id"], 862)
        self.assertEqual(len(calls), 2)

    def test_links_mapping_skips_search_and_429_retries(self):
        os.environ["TMDB_READ_TOKEN"] = "fake-bearer"
        links_file = Path(tempfile.mktemp(suffix=".csv"))
        links_file.write_text("movieId,imdbId,tmdbId\n50,0076759,11\n", encoding="utf-8")
        attempts = [0]

        def fake_open(req, timeout=5.0):
            attempts[0] += 1
            if attempts[0] == 1:
                raise urllib.error.HTTPError(req.full_url, 429, "Rate limit", {"Retry-After": "0.01"}, io.BytesIO())
            return _FakeResp({"id": 11, "overview": "A long time ago...", "credits": {"crew": [], "cast": []}})

        try:
            prov = TMDBMetadataProvider(store=self.store, links_path=links_file, opener=fake_open)
            res = prov.get_metadata(50, "Star Wars (1977)", 1977)
            self.assertEqual(res["tmdb_id"], 11)
            self.assertEqual(attempts[0], 2)
        finally:
            links_file.unlink(missing_ok=True)

    def test_proxy_optional_default_off_and_socks5_handshake(self):
        import socket as sock_mod

        old_proxy = os.environ.pop("TMDB_PROXY", None)
        try:
            # Default: zero proxy configured -> direct urlopen, canonical image URLs untouched.
            prov = TMDBMetadataProvider(store=self.store)
            self.assertIsNone(prov.proxy_url)
            self.assertFalse(prov.status_summary()["proxy_configured"])
            self.assertIs(prov._opener, urllib.request.urlopen)

            # Invalid scheme rejected cleanly.
            with self.assertRaises(ValueError):
                build_tmdb_opener("ftp://127.0.0.1:9999")

            # Loopback mock SOCKS5 server verifies RFC 1928 remote-DNS handshake (ATYP=0x03).
            srv = sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM)
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            seen = {}

            def accept_socks():
                conn, _ = srv.accept()
                try:
                    seen["greeting"] = conn.recv(3)
                    conn.sendall(b"\x05\x00")
                    hdr = conn.recv(5)
                    alen = hdr[4]
                    seen["target"] = conn.recv(alen).decode("ascii")
                    seen["port"] = int.from_bytes(conn.recv(2), "big")
                    # Reject CONNECT with general failure (0x01) after verifying handshake bytes.
                    conn.sendall(b"\x05\x01\x00\x01\x00\x00\x00\x00\x00\x00")
                finally:
                    conn.close()
                    srv.close()

            t = threading.Thread(target=accept_socks, daemon=True)
            t.start()

            os.environ["TMDB_API_KEY"] = "fake-secret-key"
            os.environ["TMDB_PROXY"] = f"socks5h://127.0.0.1:{port}"
            proxied = TMDBMetadataProvider(store=self.store, cooldown_seconds=5.0)
            self.assertTrue(proxied.status_summary()["proxy_configured"])
            res = proxied.get_metadata(1, "Toy Story (1995)", 1995)
            t.join(timeout=3.0)
            self.assertEqual(seen.get("greeting"), b"\x05\x01\x00")
            self.assertEqual(seen.get("target"), "api.themoviedb.org")
            self.assertEqual(seen.get("port"), 443)
            self.assertEqual(res["status"], "error")
            self.assertEqual(
                rewrite_tmdb_image_url("https://image.tmdb.org/t/p/w500/abc_123.jpg"),
                "/api/tmdb_image?size=w500&path=/abc_123.jpg",
            )
        finally:
            if old_proxy is not None:
                os.environ["TMDB_PROXY"] = old_proxy
            else:
                os.environ.pop("TMDB_PROXY", None)


class BrowseFiltersSecurityAndBackupTests(unittest.TestCase):
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
            return json.loads(r.read().decode("utf-8")), dict(r.headers)

    def _post(self, path, obj):
        req = urllib.request.Request(
            f"http://{self.host}:{self.port}{path}",
            data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode("utf-8"))

    def test_browse_sorting_year_and_rating_filters(self):
        top_rated, _ = self._get("/api/browse?sort=highest_rated&limit=5")
        means = [m["mean_rating"] for m in top_rated["movies"]]
        self.assertEqual(means, sorted(means, reverse=True))
        self.assertTrue(all(m["rating_count"] >= 15 for m in top_rated["movies"]))

        era, _ = self._get("/api/browse?year_min=1990&year_max=1995&min_rating=3.5&sort=year_asc&limit=10")
        self.assertTrue(era["movies"])
        self.assertTrue(all(1990 <= m["year"] <= 1995 and m["mean_rating"] >= 3.5 for m in era["movies"]))

    def test_security_headers_sql_injection_and_secrets_never_exposed(self):
        status, headers = self._get("/api/metadata/status")
        self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(headers.get("X-Frame-Options"), "SAMEORIGIN")
        self.assertNotIn("api_key", json.dumps(status).lower())

        # SQL injection in username or search never breaks tables or alters schema.
        evil = "bob'; DROP TABLE ratings;--"
        uid = self._post("/api/users", {"name": evil})["user_id"]
        self._post("/api/rate", {"user_id": uid, "movie_id": 50, "rating": 5.0})
        self.assertEqual(len(self.store.rating_rows(uid)), 1)
        sq, _ = self._get("/api/search?q=" + urllib.parse.quote("' OR 1=1--"))
        self.assertEqual(sq["results"], [])
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/api/tmdb_image?size=w500&path=/../etc/passwd")
        self.assertEqual(ctx.exception.code, 400)

    def test_concurrent_ratings_and_backup_recovery(self):
        uid = self._post("/api/users", {"name": "concurrent-u"})["user_id"]
        errors = []

        def worker(start_mid):
            try:
                for mid in range(start_mid, start_mid + 8):
                    self.store.rate(uid, mid, 4.0)
                self.store.unrate(uid, start_mid)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i * 10 + 1,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(self.store.rating_rows(uid)), 4 * 7)

        bak = Path(tempfile.mktemp(suffix=".db"))
        self.store.backup_to(bak)
        restored = UserStore(bak)
        try:
            self.assertEqual(len(restored.rating_rows(uid)), 4 * 7)
        finally:
            restored._conn.close()
            bak.unlink(missing_ok=True)


@unittest.skipUnless(HAVE_NUMPY, "experimental personality requires numpy")
class MilestoneEEngineAndPersonalityE2ETests(unittest.TestCase):
    def test_engine_divergence_and_personality_e2e_five_users(self):
        server, svc, store = create_server("ml-100k", db_path=tempfile.mktemp(suffix=".db"), port=0)
        try:
            # 1. Different users receive different recommendations.
            u_sci = store.create_user("scifi-fan")
            u_rom = store.create_user("romcom-fan")
            for mid in (50, 172, 181, 222, 228):
                store.rate(u_sci, mid, 5.0)
            for mid in (286, 255, 275, 285, 69):
                store.rate(u_rom, mid, 5.0)
            recs_sci = [r["movie_id"] for r in svc.recommend(store.user_ratings(u_sci), top_n=10)]
            recs_rom = [r["movie_id"] for r in svc.recommend(store.user_ratings(u_rom), top_n=10)]
            self.assertNotEqual(recs_sci, recs_rom)

            # 2. Seed 5 distinct profiled users with ratings & profiles; verify experimental mode E2E.
            import random
            rng = random.Random(2026)
            for idx in range(5):
                uid = store.create_user(f"pers-u-{idx}")
                store.set_personality(uid, {t: round(rng.uniform(1.0, 7.0), 1) for t in store.TRAITS})
                for mid in rng.sample(svc.catalog, 25):
                    store.rate(uid, mid, rng.choice([2.0, 4.0, 5.0]))
            # Verify persistence & isolation across users
            target_uid = store.get_user("pers-u-0")
            std_before = [r["movie_id"] for r in svc.recommend(store.user_ratings(target_uid), top_n=10)]
            exp15, meta15 = svc.experimental_recommend(store, target_uid, top_n=10, weight=0.15)
            exp80, meta80 = svc.experimental_recommend(store, target_uid, top_n=10, weight=0.80)
            self.assertEqual(meta15["profiled_users"], 5)
            self.assertNotEqual([r["movie_id"] for r in exp15], [r["movie_id"] for r in exp80])
            self.assertTrue(all(r["explanation_signal"] == "experimental_personality" for r in exp15))
            std_after = [r["movie_id"] for r in svc.recommend(store.user_ratings(target_uid), top_n=10)]
            self.assertEqual(std_before, std_after)  # standard recs strictly unaffected
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()

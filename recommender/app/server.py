"""Phase 14 web application: stdlib HTTP server + single-page UI (no new dependencies)."""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .metadata import TMDBMetadataProvider
from .service import RecommendationService
from .storage import UserStore

PERSONALITY_ENABLED = False  # Phase 13 Outcome D: experimental pathway, disabled by default
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_TYPES = {"styles.css": "text/css; charset=utf-8", "app.js": "application/javascript; charset=utf-8"}
_PAGE = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


class AppHandler(BaseHTTPRequestHandler):
    service = None
    store = None

    def log_message(self, fmt, *args):
        pass  # keep logs understandable: only errors print

    def _security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "no-referrer")

    def _send(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _page(self):
        body = (_STATIC_DIR / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name):
        ctype = _STATIC_TYPES.get(name)
        if not ctype:
            return self._send({"error": "not found"}, 404)
        body = (_STATIC_DIR / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _body_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length < 0 or length > 1_048_576:
            raise ValueError("request payload size out of bounds")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        try:
            if path == "/":
                return self._page()
            if path.startswith("/static/"):
                return self._static(path.split("/", 2)[2])
            if path == "/api/users":
                return self._send({"users": self.store.list_users(), "user_details": self.store.list_user_details()})
            if path == "/api/discover":
                uid = int(qs["user_id"][0])
                return self._send(self.service.discover(self.store.user_ratings(uid)))
            if path == "/api/browse":
                offset = max(0, int(qs.get("offset", [0])[0]))
                limit = max(1, min(100, int(qs.get("limit", [50])[0])))
                genre = qs.get("genre", [None])[0]
                sort = qs.get("sort", ["popular"])[0]
                ymin = qs.get("year_min", [None])[0]
                ymax = qs.get("year_max", [None])[0]
                mrat = qs.get("min_rating", [None])[0]
                window, total = self.service.browse(
                    offset, limit, genre, sort=sort,
                    year_min=int(ymin) if ymin else None,
                    year_max=int(ymax) if ymax else None,
                    min_rating=float(mrat) if mrat else None,
                )
                movies = self.service._enrich_details([self.service.movie_details(m) for m in window])
                return self._send({"movies": movies, "total": total})
            if path == "/api/search":
                results = self.service.search(qs.get("q", [""])[0])
                movies = self.service._enrich_details([self.service.movie_details(m) for m in results])
                return self._send({"results": movies})
            if path == "/api/metadata/status":
                return self._send(self.service.metadata.status_summary() if self.service.metadata else {"configured": False})
            if path == "/api/tmdb_image":
                if not self.service.metadata:
                    return self._send({"error": "metadata provider unavailable"}, 404)
                size = qs.get("size", ["w500"])[0]
                img_path = qs.get("path", [""])[0]
                try:
                    data, ctype = self.service.metadata.fetch_image(size, img_path)
                except ValueError as exc:
                    return self._send({"error": str(exc)}, 400)
                except Exception as exc:
                    return self._send({"error": type(exc).__name__}, 502)
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self._security_headers()
                self.end_headers()
                self.wfile.write(data)
                return
            if re.fullmatch(r"/api/movie/\d+/similar", path):
                mid = int(path.split("/")[3])
                return self._send({"similar": self.service.similar_movies(mid, limit=8)})
            if path.startswith("/api/movie/"):
                mid = int(path.rsplit("/", 1)[1])
                details = self.service.movie_details(mid, fetch_remote=True)
                return self._send(details) if details else self._send({"error": f"unknown movie {mid}"}, 404)
            if path == "/api/watchlist":
                uid = int(qs["user_id"][0])
                mids = self.store.watchlist_movies(uid)
                movies = self.service._enrich_details([self.service.movie_details(m) for m in mids])
                return self._send({"movie_ids": mids, "movies": movies})
            if path == "/api/personality/profile":
                uid = int(qs["user_id"][0])
                prof = self.store.get_personality(uid)
                return self._send({"profile": prof, "experimental_available": prof is not None})
            if path == "/api/my_ratings":
                uid = int(qs["user_id"][0])
                return self._send({"ratings": self.store.rating_rows(uid)})
            if path == "/api/recommendations":
                uid = int(qs["user_id"][0])
                top_n = max(1, min(50, int(qs.get("top_n", [10])[0])))
                exclude = qs.get("exclude_rated", ["true"])[0].lower() != "false"
                recs = self.service._enrich_details(
                    self.service.recommend(self.store.user_ratings(uid), top_n=top_n, exclude_rated=exclude))
                return self._send({"recommendations": recs, "personality_enabled": PERSONALITY_ENABLED})
            return self._send({"error": f"unknown path {path}"}, 404)
        except Exception as exc:  # keep the server alive on bad requests
            return self._send({"error": str(exc)}, 400)

    def do_POST(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        try:
            body = self._body_json()
            if parsed.path == "/api/users":
                uid = self.store.create_user(body.get("name", ""))
                return self._send({"user_id": uid})
            if parsed.path == "/api/rate":
                uid = int(body["user_id"])
                self.store.rate(uid, int(body["movie_id"]), body.get("rating"), body.get("liked"))
                return self._send({"ok": True, "rating": round(float(body["rating"]), 1)})
            if parsed.path == "/api/like":
                uid = int(body["user_id"])
                self.store.like(uid, int(body["movie_id"]), liked=bool(body.get("liked", True)))
                return self._send({"ok": True})
            if parsed.path == "/api/unrate":
                uid = int(body["user_id"])
                self.store.unrate(uid, int(body["movie_id"]))
                return self._send({"ok": True})
            if parsed.path == "/api/watchlist/toggle":
                uid = int(body["user_id"])
                saved = self.store.toggle_watchlist(uid, int(body["movie_id"]))
                return self._send({"ok": True, "saved": saved})
            if parsed.path == "/api/reset":
                uid = int(qs["user_id"][0])
                self.store.reset_user(uid)
                return self._send({"ok": True, "reset": uid})
            if parsed.path == "/api/rebuild":
                return self._send(self.service.rebuild())
            if parsed.path == "/api/personality/profile":
                body = dict(body)
                uid = int(body.pop("user_id"))
                values = self.store.set_personality(uid, body)
                return self._send({"ok": True, "profile": values,
                                   "scale": "self-reported 1-7 per trait; experimental input, not a validated assessment"})
            if parsed.path == "/api/personality/profile/delete":
                uid = int(body["user_id"])
                self.store.delete_personality(uid)
                return self._send({"ok": True, "deleted": uid})
            if parsed.path == "/api/personality":
                # Experimental pathway stub (Phase 13 Outcome D): can never affect standard recommendations.
                return self._send({"personality_enabled": PERSONALITY_ENABLED,
                                   "note": "experimental personality pathway is disabled by default and never influences standard recommendations"})
            if parsed.path == "/api/personality/recommendations":
                uid = int(qs["user_id"][0])
                weight = qs.get("weight", [None])[0]
                top_n = max(1, min(50, int(qs.get("top_n", [10])[0])))
                recs, meta = self.service.experimental_recommend(
                    self.store, uid, top_n=top_n,
                    weight=None if weight in (None, "") else float(weight))
                return self._send({"recommendations": self.service._enrich_details(recs), "meta": meta,
                                   "experimental": True, "personality_enabled": PERSONALITY_ENABLED})
            return self._send({"error": f"unknown path {parsed.path}"}, 404)
        except (ValueError, KeyError) as exc:
            return self._send({"error": str(exc)}, 400)


def create_server(dataset="ml-100k", db_path="artifacts/application/app_data.db", host="127.0.0.1", port=8000, tmdb_proxy=None):
    service = RecommendationService(dataset=dataset)
    store = UserStore(db_path)
    links_candidate = Path(f"data/raw/{dataset}/links.csv")
    cache_dir = Path(db_path).resolve().parent / "image_cache"
    service.metadata = TMDBMetadataProvider(
        store=store,
        links_path=links_candidate if links_candidate.is_file() else None,
        proxy_url=tmdb_proxy,
        image_cache_dir=cache_dir,
    )
    AppHandler.service = service
    AppHandler.store = store
    server = ThreadingHTTPServer((host, port), AppHandler)
    return server, service, store


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 14 movie recommendation application")
    parser.add_argument("--dataset", choices=("ml-100k", "ml-1m"), default="ml-100k")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tmdb-proxy", default=None, help="Optional proxy for TMDB (e.g. socks5h://127.0.0.1:10808)")
    args = parser.parse_args()
    server, service, _store = create_server(args.dataset, host=args.host, port=args.port, tmdb_proxy=args.tmdb_proxy)
    print(f"Dataset {args.dataset}: {len(service.movies)} movies, {len(service.ratings)} ratings, "
          f"loaded in {service.load_seconds}s")
    print(f"Personality pathway: {'ENABLED (experimental)' if service.personality_enabled else 'disabled (default)'}")
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()

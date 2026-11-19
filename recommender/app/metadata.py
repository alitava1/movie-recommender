"""Optional TMDB movie metadata provider with persistent SQLite caching (stdlib-only).

Never assumes MovieLens IDs equal TMDB IDs: uses an explicit links mapping when available
or TMDB /3/search/movie by normalized title + release year. Credentials come strictly from
TMDB_READ_TOKEN or TMDB_API_KEY environment variables and are never sent to the frontend.
"""

import csv
import http.client
import json
import os
import re
import socket
import ssl
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP_BASE = "https://image.tmdb.org/t/p/w1280"
_IMAGE_SIZES = {"w500", "w1280"}
_IMAGE_PATH_RE = re.compile(r"^/[A-Za-z0-9_.-]+\.(?:jpg|jpeg|png|webp)$")
_IMAGE_URL_RE = re.compile(r"^https://image\.tmdb\.org/t/p/(w500|w1280)(/[A-Za-z0-9_.-]+\.(?:jpg|jpeg|png|webp))$")
_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")
_ARTICLE_RE = re.compile(r"^(.*),\s*(The|An|A)$", re.IGNORECASE)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise OSError("unexpected EOF from SOCKS5 proxy")
        buf.extend(chunk)
    return bytes(buf)


class _Socks5HTTPSConnection(http.client.HTTPSConnection):
    """Minimal stdlib-only SOCKS5 HTTPS connection using remote hostname resolution (ATYP=0x03)."""

    def __init__(self, host, port=None, proxy_host="127.0.0.1", proxy_port=10808, **kwargs):
        super().__init__(host, port=port, **kwargs)
        self._proxy_host = proxy_host
        self._proxy_port = int(proxy_port)

    def connect(self):
        sock = socket.create_connection((self._proxy_host, self._proxy_port), self.timeout, self.source_address)
        try:
            sock.sendall(b"\x05\x01\x00")
            ver, method = _recv_exact(sock, 2)
            if ver != 5 or method != 0:
                raise OSError(f"SOCKS5 auth negotiation failed (ver={ver}, method={method})")
            host_bytes = self.host.encode("idna")
            if len(host_bytes) > 255:
                raise ValueError("hostname too long for SOCKS5")
            sock.sendall(b"\x05\x01\x00\x03" + bytes([len(host_bytes)]) + host_bytes + struct.pack("!H", self.port))
            ver, rep, _rsv, atyp = _recv_exact(sock, 4)
            if ver != 5 or rep != 0:
                raise OSError(f"SOCKS5 CONNECT failed (rep={rep})")
            if atyp == 1:
                _recv_exact(sock, 6)
            elif atyp == 3:
                alen = _recv_exact(sock, 1)[0]
                _recv_exact(sock, alen + 2)
            elif atyp == 4:
                _recv_exact(sock, 18)
            else:
                raise OSError(f"SOCKS5 unsupported address type {atyp}")
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except Exception:
            sock.close()
            raise


class _Socks5HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, proxy_host, proxy_port):
        super().__init__(context=ssl.create_default_context())
        self._proxy_host = proxy_host
        self._proxy_port = int(proxy_port)

    def https_open(self, req):
        return self.do_open(
            lambda host, **kwargs: _Socks5HTTPSConnection(
                host, proxy_host=self._proxy_host, proxy_port=self._proxy_port, **kwargs
            ),
            req,
        )


def build_tmdb_opener(proxy_url=None):
    """Build an optional proxy-aware urllib opener; defaults to direct urllib when proxy_url is empty."""
    raw = (proxy_url or "").strip()
    if not raw:
        return urllib.request.urlopen
    parsed = urllib.parse.urlparse(raw)
    scheme = parsed.scheme.lower()
    if scheme in ("socks5", "socks5h"):
        if not parsed.hostname:
            raise ValueError(f"invalid SOCKS5 proxy URL: {raw}")
        opener = urllib.request.build_opener(_Socks5HTTPSHandler(parsed.hostname, parsed.port or 1080))
        return opener.open
    if scheme in ("http", "https"):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": raw, "https": raw}))
        return opener.open
    raise ValueError(f"unsupported TMDB_PROXY scheme: {scheme}")


def redact_proxy_url(proxy_url):
    if not proxy_url:
        return None
    p = urllib.parse.urlparse(proxy_url)
    return f"{p.scheme}://{p.hostname}:{p.port}" if p.hostname and p.port else p.scheme


def rewrite_tmdb_image_url(url):
    """Convert canonical TMDB image URL to local /api/tmdb_image proxy path when proxying is enabled."""
    if not url:
        return url
    m = _IMAGE_URL_RE.match(url)
    if not m:
        return url
    return f"/api/tmdb_image?size={m.group(1)}&path={urllib.parse.quote(m.group(2), safe='/')}"


def normalize_movielens_title(title):
    """Strip trailing '(YYYY)' and move trailing ', The/An/A' to the front for TMDB search."""
    cleaned = _YEAR_RE.sub("", (title or "").strip()).strip()
    # MovieLens sometimes has alternate titles in parens, e.g. 'La Cité... (City of Lost Children, The)'
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned).strip() or cleaned
    m = _ARTICLE_RE.match(cleaned)
    if m:
        cleaned = f"{m.group(2)} {m.group(1)}".strip()
    return cleaned


def load_links_map(links_path):
    """Optional MovieLens links.csv (movieId,imdbId,tmdbId) -> {ml_id: tmdb_id}."""
    mapping = {}
    p = Path(links_path) if links_path else None
    if not p or not p.is_file():
        return mapping
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ml_id = int(row.get("movieId") or row.get("movie_id") or 0)
                tmdb_raw = (row.get("tmdbId") or row.get("tmdb_id") or "").strip()
                if ml_id and tmdb_raw:
                    mapping[ml_id] = int(tmdb_raw)
            except (TypeError, ValueError):
                continue
    return mapping


class TMDBMetadataProvider:
    """Rate-limit and timeout-safe TMDB client backed by UserStore's SQLite metadata cache."""

    def __init__(
        self,
        store=None,
        links_path=None,
        timeout=2.5,
        max_retries=2,
        opener=None,
        cooldown_seconds=60.0,
        proxy_url=None,
        image_cache_dir=None,
    ):
        self.store = store
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        env_proxy = os.environ.get("TMDB_PROXY", "").strip()
        self.proxy_url = (proxy_url if proxy_url is not None else env_proxy) or None
        self._opener = opener or build_tmdb_opener(self.proxy_url)
        self.links = load_links_map(links_path or os.environ.get("MOVIELENS_LINKS_PATH"))
        self.cooldown_seconds = float(cooldown_seconds)
        self._cooldown_until = 0.0
        self._last_error = None
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir else None

    @staticmethod
    def is_configured():
        return bool(os.environ.get("TMDB_READ_TOKEN") or os.environ.get("TMDB_API_KEY"))

    def status_summary(self):
        return {
            "provider": "TMDB",
            "configured": self.is_configured(),
            "reachable": time.monotonic() >= self._cooldown_until if self.is_configured() else False,
            "last_error": self._last_error,
            "proxy_configured": bool(self.proxy_url),
            "proxy": redact_proxy_url(self.proxy_url),
            "links_loaded": len(self.links),
            "cached_movies": self.store.metadata_count() if self.store else 0,
        }

    def fetch_image(self, size, img_path):
        """Fetch and optionally cache a validated TMDB poster/backdrop image through the configured opener."""
        if size not in _IMAGE_SIZES or not _IMAGE_PATH_RE.fullmatch(img_path or ""):
            raise ValueError("invalid TMDB image request")
        ext = img_path.rsplit(".", 1)[-1].lower()
        ctype = "image/png" if ext == "png" else ("image/webp" if ext == "webp" else "image/jpeg")
        cache_file = None
        if self.image_cache_dir is not None:
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.image_cache_dir / f"{size}_{img_path.lstrip('/')}"
            if cache_file.is_file():
                return cache_file.read_bytes(), ctype
        if time.monotonic() < self._cooldown_until:
            raise OSError(self._last_error or "TMDB image host in cooldown")
        url = f"https://image.tmdb.org/t/p/{size}{img_path}"
        req = urllib.request.Request(url, headers={"User-Agent": "MovieLens-Recommender/1.0"}, method="GET")
        try:
            with self._opener(req, timeout=max(self.timeout, 5.0)) as resp:
                data = resp.read()
        except Exception as exc:
            self._last_error = type(exc).__name__
            self._cooldown_until = time.monotonic() + self.cooldown_seconds
            raise
        if cache_file is not None and data:
            tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(cache_file)
        return data, ctype

    def _request_json(self, path, params=None):
        token = os.environ.get("TMDB_READ_TOKEN", "").strip()
        api_key = os.environ.get("TMDB_API_KEY", "").strip()
        if not token and not api_key:
            raise RuntimeError("TMDB credentials not configured")

        query = dict(params or {})
        headers = {"Accept": "application/json", "User-Agent": "MovieLens-Recommender/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            query["api_key"] = api_key

        qs = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        url = f"{TMDB_API_BASE}{path}" + (f"?{qs}" if qs else "")
        req = urllib.request.Request(url, headers=headers, method="GET")

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_err = exc
                if exc.code == 429 and attempt < self.max_retries:
                    retry_after = float(exc.headers.get("Retry-After", "1")) if exc.headers else 1.0
                    time.sleep(min(retry_after, 3.0))
                    continue
                if exc.code >= 500 and attempt < self.max_retries:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = exc
                raise
        raise last_err

    def _resolve_tmdb_id(self, movie_id, title, year):
        if movie_id in self.links:
            return self.links[movie_id]
        query = normalize_movielens_title(title)
        if not query:
            return None
        params = {"query": query, "include_adult": "false"}
        if year:
            params["year"] = str(year)
        res = self._request_json("/search/movie", params)
        results = res.get("results") or []
        if not results and year:
            # Retry once without year constraint if release year differs by +-1 between MovieLens & TMDB.
            res = self._request_json("/search/movie", {"query": query, "include_adult": "false"})
            results = res.get("results") or []
        if not results:
            return None
        if year:
            for item in results:
                rd = item.get("release_date") or ""
                if rd.startswith(str(year)):
                    return int(item["id"])
        return int(results[0]["id"])

    def _fetch_movie(self, tmdb_id):
        data = self._request_json(f"/movie/{tmdb_id}", {"append_to_response": "credits"})
        credits_doc = data.get("credits") or {}
        crew = credits_doc.get("crew") or []
        cast = credits_doc.get("cast") or []
        directors = [c["name"] for c in crew if c.get("job") == "Director" and c.get("name")]
        poster_path = data.get("poster_path")
        backdrop_path = data.get("backdrop_path")
        release_date = data.get("release_date") or ""
        return {
            "status": "found",
            "tmdb_id": int(data["id"]),
            "poster_url": f"{TMDB_POSTER_BASE}{poster_path}" if poster_path else None,
            "backdrop_url": f"{TMDB_BACKDROP_BASE}{backdrop_path}" if backdrop_path else None,
            "overview": (data.get("overview") or "").strip() or None,
            "release_date": release_date or None,
            "runtime": data.get("runtime"),
            "original_language": data.get("original_language") or None,
            "tmdb_rating": round(float(data["vote_average"]), 2) if data.get("vote_average") else None,
            "tmdb_votes": data.get("vote_count"),
            "director": directors[0] if directors else None,
            "cast": [c["name"] for c in cast[:5] if c.get("name")],
            "tmdb_genres": [g["name"] for g in (data.get("genres") or []) if g.get("name")],
        }

    def get_metadata(self, movie_id, title, year=None, allow_remote=True):
        """Return cached metadata if present; fetch from TMDB only when allowed and configured."""
        if self.store:
            cached = self.store.get_metadata(movie_id)
            if cached is not None:
                return cached
        if not allow_remote or not self.is_configured():
            return {"status": "unconfigured" if not self.is_configured() else "uncached"}
        if time.monotonic() < self._cooldown_until:
            return {"status": "error", "error": self._last_error or "unreachable_cooldown"}

        try:
            tmdb_id = self._resolve_tmdb_id(movie_id, title, year)
            if tmdb_id is None:
                payload = {"status": "not_found"}
            else:
                payload = self._fetch_movie(tmdb_id)
            self._last_error = None
            if self.store:
                self.store.set_metadata(movie_id, payload)
            return payload
        except Exception as exc:
            # Never persist transient network/rate-limit failures as not_found; enter short cooldown so UI stays responsive.
            self._last_error = type(exc).__name__
            self._cooldown_until = time.monotonic() + self.cooldown_seconds
            return {"status": "error", "error": self._last_error}

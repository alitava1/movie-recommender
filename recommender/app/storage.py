"""Phase 14 local persistence: SQLite user accounts, ratings, likes/dislikes, settings."""

import json
import sqlite3
import threading
from pathlib import Path

RATING_MIN, RATING_MAX = 0.5, 5.0
LIKE_VALUE, DISLIKE_VALUE = 5.0, 1.0
VALID_RATINGS = {round(0.5 * i, 1) for i in range(1, 11)}


class UserStore:
    """SQLite-backed per-user interaction storage; survives restarts; never touches research data."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS ratings (
                user_id INTEGER NOT NULL REFERENCES users(user_id),
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                liked INTEGER,                -- 1 like / 0 dislike / NULL neutral
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, movie_id)
            );
            CREATE TABLE IF NOT EXISTS settings (
                user_id INTEGER NOT NULL REFERENCES users(user_id),
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY (user_id, key)
            );
            CREATE TABLE IF NOT EXISTS personality (
                user_id INTEGER PRIMARY KEY REFERENCES users(user_id),
                openness REAL NOT NULL,
                conscientiousness REAL NOT NULL,
                extraversion REAL NOT NULL,
                agreeableness REAL NOT NULL,
                neuroticism REAL NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS watchlist (
                user_id INTEGER NOT NULL REFERENCES users(user_id),
                movie_id INTEGER NOT NULL,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, movie_id)
            );
            CREATE TABLE IF NOT EXISTS movie_metadata (
                movie_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    # -- users ----------------------------------------------------------------

    def create_user(self, name):
        name = (name or "").strip()
        if not name:
            raise ValueError("user name must be non-empty")
        with self._lock:
            try:
                cur = self._conn.execute("INSERT INTO users(name) VALUES (?)", (name,))
                self._conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError(f"user {name!r} already exists") from None
            return cur.lastrowid

    def get_user(self, name):
        row = self._conn.execute("SELECT user_id FROM users WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def list_users(self):
        return [r[0] for r in self._conn.execute("SELECT user_id FROM users ORDER BY user_id")]

    def list_user_details(self):
        return [{"user_id": r[0], "name": r[1]}
                for r in self._conn.execute("SELECT user_id, name FROM users ORDER BY user_id")]

    # -- ratings ----------------------------------------------------------------

    @staticmethod
    def _validate_rating(rating):
        try:
            value = round(float(rating), 1)
        except (TypeError, ValueError):
            raise ValueError(f"rating must be a number in [{RATING_MIN}, {RATING_MAX}]") from None
        if not RATING_MIN <= value <= RATING_MAX:
            raise ValueError(f"rating must be in [{RATING_MIN}, {RATING_MAX}]")
        return value

    def rate(self, user_id, movie_id, rating, liked=None):
        value = self._validate_rating(rating)
        if liked not in (None, True, False, 0, 1):
            raise ValueError("liked must be true, false, or null")
        liked_db = None if liked is None else (1 if liked in (True, 1) else 0)
        with self._lock:
            self._conn.execute(
                "INSERT INTO ratings(user_id, movie_id, rating, liked) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, movie_id) DO UPDATE SET rating = excluded.rating, "
                "liked = excluded.liked, updated_at = CURRENT_TIMESTAMP",
                (user_id, int(movie_id), value, liked_db),
            )
            self._conn.commit()

    def like(self, user_id, movie_id, liked=True):
        """Like/dislike stores an explicit preference and an effective interaction rating."""
        if liked not in (True, False):
            raise ValueError("liked must be true or false")
        value = LIKE_VALUE if liked else DISLIKE_VALUE
        self.rate(user_id, movie_id, value, liked=liked)

    def unrate(self, user_id, movie_id):
        with self._lock:
            self._conn.execute("DELETE FROM ratings WHERE user_id = ? AND movie_id = ?", (user_id, int(movie_id)))
            self._conn.commit()

    def unrated_movies(self, user_id):
        return {int(r[0]) for r in self._conn.execute(
            "SELECT movie_id FROM ratings WHERE user_id = ?", (user_id,))}

    def user_ratings(self, user_id):
        """Effective interaction history: like -> 5.0, dislike -> 1.0, else stored rating."""
        history = {}
        for mid, rating, liked in self._conn.execute(
                "SELECT movie_id, rating, liked FROM ratings WHERE user_id = ?", (user_id,)):
            history[int(mid)] = LIKE_VALUE if liked == 1 else DISLIKE_VALUE if liked == 0 else rating
        return history

    def rating_rows(self, user_id):
        return [{"movie_id": int(m), "rating": r, "liked": None if lk is None else bool(lk), "updated_at": u}
                for m, r, lk, u in self._conn.execute(
                    "SELECT movie_id, rating, liked, updated_at FROM ratings WHERE user_id = ? ORDER BY updated_at DESC",
                    (user_id,))]

    def set_setting(self, user_id, key, value):
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings(user_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
                (user_id, str(key), None if value is None else str(value)),
            )
            self._conn.commit()

    def get_setting(self, user_id, key):
        row = self._conn.execute("SELECT value FROM settings WHERE user_id = ? AND key = ?",
                                 (user_id, str(key))).fetchone()
        return row[0] if row else None

    # -- personality profiles (self-reported, experimental; Phase 15) ----------

    TRAIT_RANGE = (1.0, 7.0)
    TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

    @classmethod
    def _validate_profile(cls, profile):
        lo, hi = cls.TRAIT_RANGE
        if not isinstance(profile, dict):
            raise ValueError("profile must be a JSON object of trait scores")
        missing = [t for t in cls.TRAITS if t not in profile]
        if missing:
            raise ValueError(f"missing traits: {missing}")
        extra = [k for k in profile if k not in cls.TRAITS]
        if extra:
            raise ValueError(f"unknown trait keys: {extra}")
        out = {}
        for t in cls.TRAITS:
            try:
                v = float(profile[t])
            except (TypeError, ValueError):
                raise ValueError(f"trait {t!r} must be a number in [{lo}, {hi}]") from None
            if not lo <= v <= hi:
                raise ValueError(f"trait {t!r} must be in [{lo}, {hi}]")
            out[t] = v
        return out

    def set_personality(self, user_id, profile):
        """Create or update the user's self-reported Big Five profile (1-7 per trait)."""
        values = self._validate_profile(profile)
        with self._lock:
            self._conn.execute(
                "INSERT INTO personality(user_id, openness, conscientiousness, extraversion, "
                "agreeableness, neuroticism) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET openness = excluded.openness, "
                "conscientiousness = excluded.conscientiousness, extraversion = excluded.extraversion, "
                "agreeableness = excluded.agreeableness, neuroticism = excluded.neuroticism, "
                "updated_at = CURRENT_TIMESTAMP",
                (user_id, values["openness"], values["conscientiousness"], values["extraversion"],
                 values["agreeableness"], values["neuroticism"]),
            )
            self._conn.commit()
        return values

    def get_personality(self, user_id):
        row = self._conn.execute(
            "SELECT openness, conscientiousness, extraversion, agreeableness, neuroticism, updated_at "
            "FROM personality WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return dict(zip(self.TRAITS, row[:5])) | {"updated_at": row[5]}

    def delete_personality(self, user_id):
        with self._lock:
            self._conn.execute("DELETE FROM personality WHERE user_id = ?", (user_id,))
            self._conn.commit()

    def profiled_users(self):
        """Users who explicitly opted in with a self-reported profile."""
        return {int(r[0]) for r in self._conn.execute("SELECT user_id FROM personality")}

    # -- watchlist -------------------------------------------------------------

    def watchlist_movies(self, user_id):
        return [int(r[0]) for r in self._conn.execute(
            "SELECT movie_id FROM watchlist WHERE user_id = ? ORDER BY added_at DESC, movie_id DESC",
            (user_id,))]

    def toggle_watchlist(self, user_id, movie_id):
        mid = int(movie_id)
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM watchlist WHERE user_id = ? AND movie_id = ?", (user_id, mid)).fetchone()
            if exists:
                self._conn.execute("DELETE FROM watchlist WHERE user_id = ? AND movie_id = ?", (user_id, mid))
                self._conn.commit()
                return False
            self._conn.execute("INSERT INTO watchlist(user_id, movie_id) VALUES (?, ?)", (user_id, mid))
            self._conn.commit()
            return True

    def reset_user(self, user_id):
        """Delete all application data for one user (mission requirement)."""
        with self._lock:
            for table in ("ratings", "settings", "personality", "watchlist"):
                self._conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
            self._conn.commit()

    def delete_user(self, user_id):
        self.reset_user(user_id)
        with self._lock:
            self._conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            self._conn.commit()

    # -- persistent movie metadata cache & backups ---------------------------

    def get_metadata(self, movie_id):
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM movie_metadata WHERE movie_id = ?", (int(movie_id),)).fetchone()
        return json.loads(row[0]) if row else None

    def get_metadata_batch(self, movie_ids):
        ids = [int(m) for m in movie_ids]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT movie_id, payload_json FROM movie_metadata WHERE movie_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {int(mid): json.loads(payload) for mid, payload in rows}

    def set_metadata(self, movie_id, payload):
        status = str(payload.get("status", "unknown"))
        encoded = json.dumps(payload)
        with self._lock:
            self._conn.execute(
                "INSERT INTO movie_metadata(movie_id, status, payload_json) VALUES (?, ?, ?) "
                "ON CONFLICT(movie_id) DO UPDATE SET status = excluded.status, "
                "payload_json = excluded.payload_json, updated_at = CURRENT_TIMESTAMP",
                (int(movie_id), status, encoded),
            )
            self._conn.commit()

    def metadata_count(self):
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM movie_metadata WHERE status = 'found'").fetchone()[0])

    def backup_to(self, dest_path):
        """Atomic online SQLite backup for safe recovery."""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target = sqlite3.connect(str(dest))
            try:
                self._conn.backup(target)
            finally:
                target.close()
        return dest

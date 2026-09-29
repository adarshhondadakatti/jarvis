"""
SQLite-backed persistent memory store for JARVIS.

This module provides the database infrastructure for face recognition,
event memory, media memory, and vector storage.

Text-based memory is now stored in Open Knowledge Format (OKF) — see
memory_manager.py. The database here is used exclusively for:
  - Face recognition (people, embeddings)
  - Event memory (events, event_media, event_people)
  - Media memory (media, keyframes, detected_objects, detected_people, ocr_text, scene_descriptions)
  - Vector store (embeddings)
"""
import sqlite3
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR     = get_base_dir()
DB_PATH      = BASE_DIR / "memory" / "jarvis_memory.db"
MEDIA_DIR    = BASE_DIR / "memory" / "storage" / "media"

_lock = None  # Lazy-initialized

def _get_lock():
    global _lock
    if _lock is None:
        from threading import Lock
        _lock = Lock()
    return _lock


# ── Connection helper ──────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """Open a connection with row factory and foreign keys enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create the schema if it doesn't exist. Called once at startup."""
    with _get_lock():
        conn = _connect()
        try:
            # ── People (face memory) ──────────────────────────────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    name          TEXT NOT NULL UNIQUE,
                    first_seen    TEXT NOT NULL,
                    last_seen     TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    profile       TEXT DEFAULT 'default',
                    metadata      TEXT,
                    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_people_name ON people(name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_people_profile ON people(profile)"
            )

            # ── Person embeddings (multiple per person, float32 BLOBs) ────────
            # embedding_norm stores the raw L2 norm of the pre-normalised embedding,
            # which serves as an image-quality proxy (AdaFace, CVPR 2022).
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS person_embeddings (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    embedding       BLOB NOT NULL,
                    embedding_norm  REAL DEFAULT 0.0,
                    source          TEXT,
                    confidence      REAL,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pe_person  ON person_embeddings(person_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pe_created ON person_embeddings(created_at)"
            )

            # ── Backward-compatible migration for existing databases ─────────
            # Add embedding_norm column if it doesn't exist (older schemas).
            cols = [row[1] for row in conn.execute("PRAGMA table_info(person_embeddings)").fetchall()]
            if "embedding_norm" not in cols:
                conn.execute("ALTER TABLE person_embeddings ADD COLUMN embedding_norm REAL DEFAULT 0.0")
            # ────────────────────────────────────────────────────────────────

            # ── Media files (images, videos, clips, frames) ──────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS media (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_type TEXT NOT NULL,
                    file_path  TEXT NOT NULL,
                    duration   REAL,
                    width      INTEGER,
                    height     INTEGER,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    source     TEXT,
                    sha256     TEXT,
                    metadata   TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_type   ON media(media_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_sha256 ON media(sha256)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_created ON media(created_at)"
            )

            # ── Keyframes extracted from videos ──────────────────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS keyframes (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id   INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
                    media_time REAL NOT NULL,
                    file_path  TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kf_media ON keyframes(media_id)"
            )

            # ── Detected objects in media/keyframes ──────────────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detected_objects (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id    INTEGER REFERENCES media(id) ON DELETE CASCADE,
                    keyframe_id INTEGER REFERENCES keyframes(id) ON DELETE CASCADE,
                    label       TEXT NOT NULL,
                    confidence  REAL,
                    bbox        TEXT,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_do_media  ON detected_objects(media_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_do_label  ON detected_objects(label)"
            )

            # ── Detected people in media/keyframes ───────────────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detected_people (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id    INTEGER REFERENCES media(id) ON DELETE CASCADE,
                    keyframe_id INTEGER REFERENCES keyframes(id) ON DELETE CASCADE,
                    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    confidence  REAL,
                    bbox        TEXT,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dp_media   ON detected_people(media_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dp_person  ON detected_people(person_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dp_created ON detected_people(created_at)"
            )

            # ── OCR text extracted from media ────────────────────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_text (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id    INTEGER REFERENCES media(id) ON DELETE CASCADE,
                    keyframe_id INTEGER REFERENCES keyframes(id) ON DELETE CASCADE,
                    text        TEXT NOT NULL,
                    bbox        TEXT,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ocr_media ON ocr_text(media_id)"
            )

            # ── Scene descriptions ───────────────────────────────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scene_descriptions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id    INTEGER REFERENCES media(id) ON DELETE CASCADE,
                    keyframe_id INTEGER REFERENCES keyframes(id) ON DELETE CASCADE,
                    description TEXT NOT NULL,
                    model       TEXT,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sd_media ON scene_descriptions(media_id)"
            )

            # ── Episodic memory events ───────────────────────────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type    TEXT NOT NULL,
                    title         TEXT NOT NULL,
                    description   TEXT,
                    started_at    TEXT NOT NULL,
                    ended_at      TEXT,
                    location      TEXT,
                    metadata      TEXT,
                    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type    ON events(event_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_started ON events(started_at)"
            )

            # ── Junction: events ↔ media ─────────────────────────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_media (
                    event_id  INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                    media_id  INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
                    role      TEXT,
                    PRIMARY KEY (event_id, media_id)
                )
                """
            )

            # ── Junction: events ↔ people ────────────────────────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_people (
                    event_id  INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    role      TEXT,
                    PRIMARY KEY (event_id, person_id)
                )
                """
            )

            conn.commit()
        finally:
            conn.close()


# ── Startup initializer ────────────────────────────────────────────────────────

def ensure_db_ready() -> None:
    """
    Call this at application startup.
    Creates the schema and ensures the media storage directory exists.
    """
    init_db()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

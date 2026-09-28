"""
file_finder.py — JARVIS unified file search

Replaces the two divergent implementations that used to live in
file_controller.py:
  - _find_file_by_name()  (depth-limited os.walk fallback)
  - find_files()          (unbounded rglob("*") — the actual 'find' tool action)

Both are gone. This module is the single source of truth for "where is
file X":
  - Maintains a lightweight SQLite index (path, name, ext, parent, size, mtime)
  - Excludes noisy directories up front (node_modules, .git, venvs, caches...)
    so os.walk never even descends into them, instead of scanning them and
    discarding results after the fact
  - Refreshes in the background on a timer, so search() itself never blocks
    on a full disk walk
  - Uses rapidfuzz for fuzzy name matching + ranking (falls back to plain
    substring matching if rapidfuzz isn't installed)
"""

import os
import sqlite3
import threading
import time
from pathlib import Path

try:
    from rapidfuzz import fuzz, process as _fuzz_process
    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False

# ── Config ───────────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "file_index.db"

_EXCLUDE_DIR_NAMES = {
    "node_modules", ".git", "__pycache__", "venv", ".venv", "env",
    "site-packages", "dist", "build", ".idea", ".vscode",
    "AppData", "Temp", "tmp", "$Recycle.Bin", "System Volume Information",
    ".cache", ".npm", ".gradle",
}

_EXCLUDE_EXT = {".tmp", ".lock", ".log"}

_REFRESH_INTERVAL_SEC = 15 * 60  # background rescan every 15 minutes

_last_refresh = 0.0
_refresh_thread_started = False


def _default_roots() -> list[Path]:
    home = Path.home()
    candidates = [
        home / "Desktop", home / "Downloads", home / "Documents",
        home / "Pictures", home / "Music", home / "Videos",
        home,
    ]
    seen, roots = set(), []
    for c in candidates:
        try:
            r = c.resolve()
            if r.exists() and r not in seen:
                seen.add(r)
                roots.append(c)
        except Exception:
            pass
    return roots


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    # WAL mode lets search() read while build_index() is still writing in
    # the background — without this, a search during an active rebuild can
    # hit "database is locked" and fail silently.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            path       TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            ext        TEXT,
            parent     TEXT,
            size       INTEGER,
            mtime      REAL,
            indexed_at REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON files(name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mtime ON files(mtime)")
    return conn


def build_index(roots: list[Path] = None, max_depth: int = 12, max_files: int = 200_000) -> dict:
    """
    Full (re)scan of the given roots into the SQLite index. Safe to call
    repeatedly — upserts rows. Prunes excluded directories in-place so
    os.walk never descends into them (unlike the old rglob-then-discard
    approach).
    """
    roots = roots or _default_roots()
    conn = _connect()
    cur = conn.cursor()

    count = 0
    t0 = time.time()

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        root_str = str(root.resolve())
        for dirpath, dirnames, filenames in os.walk(root_str):
            rel = os.path.relpath(dirpath, root_str)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth >= max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [
                d for d in dirnames
                if d not in _EXCLUDE_DIR_NAMES and not d.startswith(".")
            ]

            for fname in filenames:
                if count >= max_files:
                    break
                ext = Path(fname).suffix.lower()
                if ext in _EXCLUDE_EXT:
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    st = os.stat(fpath)
                except OSError:
                    continue
                cur.execute(
                    "INSERT OR REPLACE INTO files "
                    "(path, name, ext, parent, size, mtime, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (fpath, fname, ext, dirpath, st.st_size, st.st_mtime, time.time())
                )
                count += 1
                if count % 1000 == 0:
                    conn.commit()  # periodic commit — keeps write transactions short
            if count >= max_files:
                break

    conn.commit()
    conn.close()

    global _last_refresh
    _last_refresh = time.time()
    return {"files_indexed": count, "seconds": round(time.time() - t0, 2)}


def _ensure_fresh(max_age_sec: int = _REFRESH_INTERVAL_SEC):
    global _last_refresh
    if not _DB_PATH.exists():
        build_index()
        return
    if time.time() - _last_refresh > max_age_sec:
        threading.Thread(target=build_index, daemon=True).start()
        _last_refresh = time.time()  # avoid re-triggering every call while it runs


def start_background_refresh():
    """Call once at JARVIS startup for periodic rescans without blocking anything."""
    global _refresh_thread_started
    if _refresh_thread_started:
        return
    _refresh_thread_started = True

    def _loop():
        while True:
            try:
                build_index()
            except Exception as e:
                print(f"[file_finder] background index refresh failed: {e}")
            time.sleep(_REFRESH_INTERVAL_SEC)

    threading.Thread(target=_loop, daemon=True).start()


def search(query: str, extension: str = "", top_k: int = 10, min_score: int = 55) -> list[dict]:
    """
    Fuzzy search the index for files whose name best matches *query*.
    Returns [{path, name, parent, size, mtime, score}, ...], best-first.
    """
    _ensure_fresh()
    conn = _connect()
    cur = conn.cursor()

    if extension:
        ext = extension if extension.startswith(".") else f".{extension}"
        rows = cur.execute(
            "SELECT path, name, ext, parent, size, mtime FROM files WHERE ext = ?", (ext,)
        ).fetchall()
    else:
        rows = cur.execute("SELECT path, name, ext, parent, size, mtime FROM files").fetchall()
    conn.close()

    if not rows:
        return []

    if not query:
        results = [
            {"path": r[0], "name": r[1], "parent": r[3], "size": r[4], "mtime": r[5], "score": 100}
            for r in rows
        ]
        results.sort(key=lambda r: -r["mtime"])
        return results[:top_k]

    if _RAPIDFUZZ:
        names = [r[1] for r in rows]
        matches = _fuzz_process.extract(query, names, scorer=fuzz.WRatio, limit=top_k * 3)
        results, seen_paths = [], set()
        for name, score, idx in matches:
            if score < min_score:
                continue
            row = rows[idx]
            if row[0] in seen_paths:
                continue
            seen_paths.add(row[0])
            results.append({
                "path": row[0], "name": row[1], "parent": row[3],
                "size": row[4], "mtime": row[5], "score": score,
            })
        results.sort(key=lambda r: (-r["score"], -r["mtime"]))
        return results[:top_k]
    else:
        q = query.lower()
        results = [
            {"path": r[0], "name": r[1], "parent": r[3], "size": r[4], "mtime": r[5], "score": 100}
            for r in rows if q in r[1].lower()
        ]
        results.sort(key=lambda r: -r["mtime"])
        return results[:top_k]


def _format_size(size: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def find_files(name: str = "", extension: str = "", max_results: int = 10) -> str:
    """Human-readable search result — used by the tool entry point below."""
    if not name and not extension:
        return "Please provide a file name or extension to search for."

    results = search(query=name, extension=extension, top_k=max_results)
    if not results:
        query_desc = name or extension or "files"
        return f"No files matching '{query_desc}' found in the index."

    lines = [f"Found {len(results)} match(es):"]
    for r in results:
        lines.append(
            f"  📄 {r['name']}  ({_format_size(r['size'])})  — {r['parent']}  [match {r['score']:.0f}%]"
        )
    return "\n".join(lines)


def find_best_match(name: str) -> Path | None:
    """
    Drop-in replacement for the old _find_file_by_name(): returns the single
    best-matching Path, or None. file_processor.py should call this when a
    given file_path doesn't exist and needs to be resolved by name.
    """
    results = search(query=name, top_k=1)
    if results and results[0]["score"] >= 70:
        return Path(results[0]["path"])
    return None


# ── Tool entry point — register this as its OWN tool in main.py,        ──
# ── separate from file_controller and file_processor.                    ──

def file_finder(parameters: dict = None, player=None) -> str:
    params = parameters or {}
    name = params.get("name", "") or params.get("query", "")
    extension = params.get("extension", "")
    max_results = min(int(params.get("max_results", 10)), 50)

    if player:
        player.write_log(f"[file_finder] query='{name}' ext='{extension}'")

    return find_files(name=name, extension=extension, max_results=max_results)
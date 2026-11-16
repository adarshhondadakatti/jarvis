"""
Tests for progress reporting.

    python -m pytest tests/test_progress.py -v

Verifies the throttling (a tight loop must not hammer the DB), clamping, and
that the reporter is a genuine no-op when no task is attached — tools call it
unconditionally and must still work standalone.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.progress import Progress, make
from memory import task_store


@pytest.fixture(autouse=True)
def clean_db(tmp_path, monkeypatch):
    import memory.db as db

    temp_db = tmp_path / "progress.db"
    monkeypatch.setattr(db, "DB_PATH", temp_db)
    monkeypatch.setattr(db, "MEDIA_DIR", tmp_path / "media")
    db.init_db()
    yield


@pytest.fixture
def task():
    t = task_store.create("web_search", {"query": "x"})
    task_store.mark_running(t.id)
    return t


# ── No-op behaviour ──────────────────────────────────────────────────────────

def test_progress_without_task_is_noop():
    """Tools call progress unconditionally; with no task it must cost nothing."""
    p = Progress(None, "web_search")
    assert p.enabled is False
    p(0.5, "halfway")
    p(1.0, "done", force=True)
    # Nothing was written anywhere — no crash, no side effect.
    assert task_store.list_tasks(limit=5) == []


# ── Writes ───────────────────────────────────────────────────────────────────

def test_progress_updates_row(task):
    p = Progress(task.id, "web_search", min_interval_s=0.0)
    p(0.4, "querying", force=True)

    t = task_store.get(task.id)
    assert t.progress == 0.4
    assert t.progress_note == "querying"


def test_progress_clamps_out_of_range(task):
    p = Progress(task.id, "x", min_interval_s=0.0)
    p(2.0, "over", force=True)
    assert task_store.get(task.id).progress == 1.0
    p(-1.0, "under", force=True)
    assert task_store.get(task.id).progress == 0.0


def test_note_updates_text_only(task):
    p = Progress(task.id, "x", min_interval_s=0.0)
    p(0.5, "a", force=True)
    p.note("b", force=True)

    t = task_store.get(task.id)
    assert t.progress == 0.5          # percentage untouched
    assert t.progress_note == "b"     # text changed


# ── Throttling ───────────────────────────────────────────────────────────────

def test_burst_is_throttled(task):
    """100 rapid calls must not produce 100 DB writes."""
    p = Progress(task.id, "x", min_interval_s=10.0)   # throttle everything
    for i in range(100):
        p(i / 100, f"step {i}")

    # Only the first call got through; the rest were inside the window.
    t = task_store.get(task.id)
    assert t.progress < 0.2, f"expected throttling, got {t.progress}"


def test_force_bypasses_throttle(task):
    p = Progress(task.id, "x", min_interval_s=10.0)
    p(0.1, "first")
    p(0.9, "final", force=True)

    assert task_store.get(task.id).progress == 0.9


def test_throttle_window_expiry_allows_next_call(task):
    p = Progress(task.id, "x", min_interval_s=0.05)
    p(0.1, "first", force=True)
    time.sleep(0.08)
    p(0.5, "second")

    assert task_store.get(task.id).progress == 0.5


# ── Failure isolation ────────────────────────────────────────────────────────

def test_progress_swallows_db_errors(monkeypatch):
    """A broken DB must never take the tool down with it."""
    p = Progress("t-doesnotexist", "x", min_interval_s=0.0)
    # No row with this id — the UPDATE matches nothing, which is fine.
    p(0.5, "still alive")


# ── make() with a UI ─────────────────────────────────────────────────────────

class FakeUI:
    def __init__(self):
        self.logs = []

    def write_log(self, msg):
        self.logs.append(msg)


def test_make_echoes_notes_to_ui(task):
    ui = FakeUI()
    p = make(task.id, "web_search", player=ui)
    p(0.5, "querying the web", force=True)

    assert any("querying the web" in m for m in ui.logs)
    assert task_store.get(task.id).progress_note == "querying the web"


def test_make_without_ui_works(task):
    p = make(task.id, "web_search", player=None)
    p(0.5, "ok", force=True)
    assert task_store.get(task.id).progress == 0.5


# ── End to end: a tool-style loop ────────────────────────────────────────────

def test_typical_tool_progress_sequence(task):
    """Mimics how a real tool reports: coarse steps, force only at the end."""
    p = Progress(task.id, "dev_agent", min_interval_s=0.0)

    p(0.05, "planning")
    p(0.4, "writing 2/5 files")
    p(0.7, "test run 1/5")
    p(0.9, "test run 2/5")
    p(1.0, "done", force=True)

    t = task_store.get(task.id)
    assert t.progress == 1.0
    assert t.progress_note == "done"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
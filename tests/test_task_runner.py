"""
Tests for concurrent tool execution and durable task tracking.

    python -m pytest tests/test_task_runner.py -v

No network, no models, no Qt — pure logic against a throwaway SQLite DB.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.task_runner import TaskRunner, describe_batch, run_batch
from memory import task_store


# ── Fixtures ─────────────────────────────────────────────────────────────────

class FakeFC:
    """Stands in for a google.genai FunctionCall."""

    def __init__(self, name: str, args: dict | None = None):
        self.name = name
        self.args = args or {}
        self.id = f"{name}-1"

    def __repr__(self) -> str:
        return f"<FakeFC {self.name}>"


@pytest.fixture(autouse=True)
def clean_db(tmp_path, monkeypatch):
    """Point the DB at a temp file and start from an empty schema."""
    import memory.db as db

    temp_db = tmp_path / "tasks.db"
    monkeypatch.setattr(db, "DB_PATH", temp_db)
    monkeypatch.setattr(db, "MEDIA_DIR", tmp_path / "media")
    db.init_db()
    yield
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(temp_db) + suffix)
        if p.exists():
            p.unlink()


@pytest.fixture
def runner():
    return TaskRunner()


# ── Task store ───────────────────────────────────────────────────────────────

def test_create_and_get():
    t = task_store.create("weather_report", {"city": "Trivandrum"})
    assert t.id.startswith("t-")
    assert t.status == task_store.QUEUED
    assert t.progress == 0.0

    got = task_store.get(t.id)
    assert got is not None
    assert got.tool == "weather_report"
    assert got.args == {"city": "Trivandrum"}


def test_lifecycle_transitions():
    t = task_store.create("code_helper", {"language": "python"})

    task_store.mark_running(t.id)
    running = task_store.get(t.id)
    assert running.status == task_store.RUNNING
    assert running.started_at is not None
    assert running.finished_at is None

    task_store.update(t.id, progress=0.5, progress_note="halfway")
    mid = task_store.get(t.id)
    assert mid.progress == 0.5
    assert mid.progress_note == "halfway"

    task_store.update(t.id, status=task_store.DONE, result="wrote main.py")
    done = task_store.get(t.id)
    assert done.status == task_store.DONE
    assert done.finished_at is not None
    assert done.duration_s is not None and done.duration_s >= 0


def test_progress_is_clamped():
    t = task_store.create("x")
    task_store.update(t.id, progress=5.0)
    assert task_store.get(t.id).progress == 1.0
    task_store.update(t.id, progress=-3.0)
    assert task_store.get(t.id).progress == 0.0


def test_failed_task_keeps_error():
    t = task_store.create("dev_agent")
    task_store.update(t.id, status=task_store.FAILED, error="no module named foo")
    got = task_store.get(t.id)
    assert got.status == task_store.FAILED
    assert "foo" in got.error
    assert got.finished_at is not None


def test_active_tasks_excludes_finished():
    a = task_store.create("a")
    b = task_store.create("b")
    c = task_store.create("c")
    task_store.mark_running(a.id)
    task_store.update(c.id, status=task_store.DONE, result="ok")

    active = task_store.active_tasks()
    assert {t.id for t in active} == {a.id, b.id}


def test_list_tasks_newest_first():
    first = task_store.create("one")
    second = task_store.create("two")
    listed = task_store.list_tasks(limit=10)
    assert listed[0].id == second.id
    assert listed[1].id == first.id


def test_status_report_mentions_live_work():
    t = task_store.create("web_search", {"query": "gemini live api"})
    task_store.mark_running(t.id)
    task_store.update(t.id, progress=0.3, progress_note="fetching")

    report = task_store.status_report()
    assert "In progress" in report
    assert "web_search" in report
    assert "30%" in report


def test_status_report_when_idle():
    assert "No tasks recorded" in task_store.status_report()


def test_purge_finished_keeps_active():
    live = task_store.create("live")
    task_store.mark_running(live.id)
    for _ in range(5):
        old = task_store.create("done")
        task_store.update(old.id, status=task_store.DONE, result="x")

    removed = task_store.purge_finished(keep=2)
    assert removed >= 1
    assert task_store.get(live.id) is not None
    assert task_store.get(live.id).status == task_store.RUNNING


# ── run_batch ────────────────────────────────────────────────────────────────

def test_run_batch_empty():
    assert asyncio.run(run_batch([], _noop)) == []


def test_run_batch_preserves_order():
    """Gemini matches responses to calls by position — order must not drift."""
    async def execute(fc):
        # Later calls finish first on purpose.
        await asyncio.sleep(0.05 - 0.01 * int(fc.name[-1]))
        return fc.name

    calls = [FakeFC("tool1"), FakeFC("tool2"), FakeFC("tool3")]
    out = asyncio.run(run_batch(calls, execute))
    assert out == ["tool1", "tool2", "tool3"]


def test_run_batch_is_actually_concurrent():
    """The whole point: 3 x 0.2s of work must not take 0.6s."""
    async def execute(fc):
        await asyncio.sleep(0.2)
        return fc.name

    calls = [FakeFC("a"), FakeFC("b"), FakeFC("c")]
    t0 = time.monotonic()
    out = asyncio.run(run_batch(calls, execute))
    elapsed = time.monotonic() - t0

    assert out == ["a", "b", "c"]
    # Generous ceiling: serial would be ~0.6s, concurrent ~0.2s.
    assert elapsed < 0.45, f"batch took {elapsed:.2f}s — looks serial"


def test_run_batch_single_call_skips_gather():
    async def execute(fc):
        return fc.name

    out = asyncio.run(run_batch([FakeFC("solo")], execute))
    assert out == ["solo"]


def test_run_batch_isolates_failures():
    """One broken tool must not take down its siblings."""
    async def execute(fc):
        if fc.name == "bad":
            raise RuntimeError("boom")
        return fc.name

    calls = [FakeFC("good1"), FakeFC("bad"), FakeFC("good2")]
    out = asyncio.run(run_batch(calls, execute))

    assert len(out) == 3
    assert out[0] == "good1"
    assert "boom" in out[1]
    assert out[2] == "good2"


def test_describe_batch():
    calls = [FakeFC("open_app"), FakeFC("email")]
    assert describe_batch(calls) == "open_app + email"


async def _noop(fc):
    return None


# ── TaskRunner integration ───────────────────────────────────────────────────

def test_runner_tracks_live_work(runner):
    rt = runner.register("browser_control", {"url": "example.com"})
    runner.start(rt)

    assert runner.is_busy()
    assert len(runner.live_tasks()) == 1

    runner.progress(rt.task_id, 0.6, "page loaded")
    assert task_store.get(rt.task_id).progress == 0.6

    runner.finish(rt, result="opened")
    assert not runner.is_busy()

    done = task_store.get(rt.task_id)
    assert done.status == task_store.DONE
    assert done.result == "opened"


def test_runner_records_failure(runner):
    rt = runner.register("dev_agent")
    runner.start(rt)
    runner.finish(rt, error="build failed: syntax error")

    failed = task_store.get(rt.task_id)
    assert failed.status == task_store.FAILED
    assert "syntax error" in failed.error
    assert not runner.is_busy()


def test_runner_register_with_explicit_id(runner):
    rt = runner.register("reminder", task_id="t-fixed123")
    assert rt.task_id == "t-fixed123"
    assert task_store.get("t-fixed123") is not None


def test_runner_count_and_report(runner):
    rt = runner.register("open_app")
    runner.start(rt)
    runner.finish(rt, result="opened")

    counts = runner.counts()
    assert counts.get(task_store.DONE) == 1
    assert "open_app" in runner.report()


def test_runner_cancel_unknown_task(runner):
    assert runner.cancel("t-nope") is False


def test_runner_cancel_live_handle(runner):
    rt = runner.register("game_updater")
    runner.start(rt)

    async def scenario():
        holder = asyncio.ensure_future(asyncio.sleep(30))
        runner.attach(rt, holder)
        return runner.cancel(rt.task_id)

    cancelled = asyncio.run(scenario())
    assert cancelled is True
    assert task_store.get(rt.task_id).status == task_store.CANCELLED


def test_runner_two_tools_both_tracked(runner):
    a = runner.register("email", {"action": "fetch"})
    b = runner.register("web_search", {"query": "news"})
    runner.start(a)
    runner.start(b)

    assert {t.tool for t in runner.live_tasks()} == {"email", "web_search"}

    runner.finish(a, result="5 unread")
    runner.finish(b, result="3 results")

    assert not runner.is_busy()
    assert len(task_store.list_tasks(limit=10)) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
core/task_runner.py — concurrent tool execution for JARVIS.

The problem this solves
-----------------------
``JarvisLive._execute_tool`` used to ``await`` every function call in a
``tool_call`` batch one after another, and the Gemini Live receive loop was
blocked for the whole time. Ask for two things — "open Chrome and check my
email" — and the second waited for the first to fully finish. That is serial
execution wearing an async costume.

What this adds
--------------
1. ``run_batch()`` — executes the function calls of a single ``tool_call``
   **concurrently**, returning their responses in the original order. Gemini
   requires the function responses to match the order of the calls it made,
   so order is preserved while the work overlaps.
2. ``TaskRunner`` — a registry of in-flight work with cancellation and
   progress, backed by ``memory.task_store`` so the state survives and is
   visible to the UI and to the assistant.

Every tool call gets a task row (queued → running → done/failed). Long-running
tools can report progress via ``TaskRunner.progress()``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

from memory import task_store

log = logging.getLogger("jarvis.task_runner")


# ── Batch execution ──────────────────────────────────────────────────────────

async def run_batch(
    calls: Sequence[Any],
    execute: Callable[[Any], Awaitable[Any]],
    *,
    preserve_order: bool = True,
) -> list[Any]:
    """
    Run ``execute(fc)`` for every function call in ``calls`` concurrently.

    ``calls`` is the ``response.tool_call.function_calls`` sequence from the
    Live API. ``execute`` is the per-call coroutine (in JARVIS:
    ``self._execute_tool``).

    Exceptions are captured per-call and returned in place of a result, so one
    failing tool cannot sink the others — the model still gets a response for
    every call it made. With ``preserve_order=False`` results arrive in
    completion order instead (useful for progress reporting, not for the
    Live API which needs matching order).

    A single call runs directly (no task overhead). Two or more are gathered,
    which is where the wall-clock saving comes from.
    """
    calls = list(calls)
    if not calls:
        return []
    if len(calls) == 1:
        return [await execute(calls[0])]

    async def _safe(fc: Any) -> Any:
        try:
            return await execute(fc)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - reported to the model as text
            log.exception("[TaskRunner] tool %s failed", getattr(fc, "name", "?"))
            return f"Tool error: {e}"

    if preserve_order:
        return list(await asyncio.gather(*(_safe(fc) for fc in calls)))

    tasks = [asyncio.ensure_future(_safe(fc)) for fc in calls]
    out: list[Any] = []
    for coro in asyncio.as_completed(tasks):
        out.append(await coro)
    return out


def describe_batch(calls: Sequence[Any]) -> str:
    """``open_app + weather_report`` — used in logs and the UI status line."""
    names = [getattr(fc, "name", "?") for fc in calls]
    return " + ".join(names)


# ── In-flight task registry ──────────────────────────────────────────────────

@dataclass
class RunningTask:
    task_id: str
    tool: str
    args: dict = field(default_factory=dict)
    started_at: float = field(default_factory=lambda: __import__("time").monotonic())
    status: str = task_store.QUEUED
    _cancel: Optional[Callable[[], None]] = None

    @property
    def elapsed(self) -> float:
        import time
        return time.monotonic() - self.started_at


class TaskRunner:
    """
    Tracks concurrently-running tool calls.

    One instance per ``JarvisLive`` session. It is deliberately thin: the
    durable record lives in SQLite via ``task_store``; this class holds the
    live handles (asyncio tasks, cancellation) that cannot be serialised.
    """

    def __init__(self) -> None:
        self._live: dict[str, RunningTask] = {}
        self._handles: dict[str, asyncio.Task] = {}
        self._lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────────

    def register(
        self,
        tool: str,
        args: Optional[dict] = None,
        *,
        task_id: Optional[str] = None,
    ) -> RunningTask:
        """Create a task row and a live handle. Call before the work starts."""
        row = task_store.create(tool, args, task_id=task_id)
        rt = RunningTask(task_id=row.id, tool=tool, args=row.args)
        with self._lock:
            self._live[row.id] = rt
        return rt

    def attach(self, rt: RunningTask, handle: asyncio.Task) -> None:
        """Bind an asyncio task so it can be cancelled."""
        with self._lock:
            self._handles[rt.task_id] = handle
            self._live[rt.task_id] = rt

    def start(self, rt: RunningTask) -> RunningTask:
        task_store.mark_running(rt.task_id)
        rt.status = task_store.RUNNING
        return rt

    def progress(
        self,
        task_id: str,
        value: float,
        note: Optional[str] = None,
    ) -> None:
        """Report progress for a task. Cheap; safe to call in a loop."""
        task_store.update(task_id, progress=value, progress_note=note)

    def finish(
        self,
        rt: RunningTask,
        *,
        status: Optional[str] = None,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Close out a task and forget its live handle."""
        with self._lock:
            self._live.pop(rt.task_id, None)
            self._handles.pop(rt.task_id, None)
        task_store.update(
            rt.task_id,
            status=status or (task_store.FAILED if error else task_store.DONE),
            progress=1.0 if not error else None,
            result=result,
            error=error,
        )

    # ── queries ─────────────────────────────────────────────────────────────

    def live_tasks(self) -> list[RunningTask]:
        with self._lock:
            return sorted(self._live.values(), key=lambda t: t.started_at)

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._live)

    def counts(self) -> dict[str, int]:
        return task_store.count_by_status()

    def report(self, limit: int = 10) -> str:
        """Spoken/printed summary of what the assistant is doing."""
        return task_store.status_report(limit=limit)

    # ── cancellation ────────────────────────────────────────────────────────

    def cancel(self, task_id: str) -> bool:
        """Cancel one in-flight task. Returns True if a handle was found."""
        with self._lock:
            handle = self._handles.get(task_id)
        if not handle or handle.done():
            return False
        handle.cancel()
        task_store.update(task_id, status=task_store.CANCELLED, error="Cancelled by user")
        return True

    def cancel_all(self) -> int:
        with self._lock:
            handles = [h for h in self._handles.values() if not h.done()]
        for h in handles:
            h.cancel()
        return len(handles)

    def clear_finished(self, keep: int = 200) -> int:
        """Prune old finished rows from the DB."""
        return task_store.purge_finished(keep=keep)


__all__ = ["run_batch", "describe_batch", "TaskRunner", "RunningTask"]
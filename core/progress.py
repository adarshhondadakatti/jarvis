"""
core/progress.py — tiny helper for reporting tool progress.

Tools run in a thread pool executor (see ``core.task_runner``), while the task
registry is updated from the asyncio side. This bridges the two: hand a tool a
``Progress`` object, it calls ``p(0.4, "extracting text")`` as it goes, and the
task row updates — which is what makes ``task_status`` and the UI able to show
a percentage instead of "0% ... 0% ... 100%".

Usage in a tool::

    def web_search(parameters, player=None, speak=None, progress=None):
        p = Progress(progress, "web_search")
        p(0.1, "querying")
        ...
        p(0.9, "formatting")

``progress`` is the ``Progress``'s optional owner (a ``TaskRunner``); when it is
absent every call is a no-op, so tools stay usable standalone from a REPL.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("jarvis.progress")

# Progress is reported from worker threads; coalesce bursts so a tight loop
# doesn't hammer SQLite (and the UI) with hundreds of writes a second.
_MIN_INTERVAL_S = 0.35


class Progress:
    """A callable progress reporter. Cheap when unused."""

    def __init__(
        self,
        task_id: Optional[str] = None,
        tool: str = "",
        *,
        min_interval_s: float = _MIN_INTERVAL_S,
    ) -> None:
        self.task_id = task_id
        self.tool = tool or "task"
        self._last_value = -1.0
        self._last_note: Optional[str] = None
        self._min_interval = min_interval_s
        self._last_at = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.task_id)

    def __call__(
        self,
        value: Optional[float] = None,
        note: Optional[str] = None,
        *,
        force: bool = False,
    ) -> None:
        """
        Record progress. ``value`` is 0.0–1.0 (clamped).

        Throttled: repeated calls with the same note and a value within a
        hair of the last one are dropped, unless ``force=True``. The final
        report from a tool should pass ``force=True`` so it always lands.
        """
        if not self.task_id:
            return

        if value is not None:
            value = max(0.0, min(1.0, float(value)))

        if not force:
            import time
            now = time.monotonic()
            if now - self._last_at < self._min_interval:
                return
            if note is None and value is not None and abs(value - self._last_value) < 0.01:
                return
            self._last_at = now

        if value is not None:
            self._last_value = value
        if note is not None:
            self._last_note = note

        try:
            from memory import task_store
            task_store.update(
                self.task_id,
                progress=value if value is not None else None,
                progress_note=note,
            )
        except Exception as e:  # never let progress reporting break the tool
            log.debug(f"[Progress] update failed for {self.task_id}: {e}")

    def note(self, text: str, *, force: bool = False) -> None:
        """Update only the human-readable status line, leaving the percentage."""
        self(value=None, note=text, force=force)


class _UiProgress(Progress):
    """A Progress that also echoes notes to a UI log, keeping the two in sync."""

    def __init__(self, task_id, tool, player, *, min_interval_s: float = _MIN_INTERVAL_S):
        super().__init__(task_id, tool, min_interval_s=min_interval_s)
        self._player = player

    def __call__(self, value=None, note=None, *, force: bool = False) -> None:
        super().__call__(value, note, force=force)
        if note and self._player is not None:
            try:
                self._player.write_log(f"  · {note}")
            except Exception:
                pass


def make(
    task_id: Optional[str],
    tool: str,
    *,
    player: Any = None,
) -> Progress:
    """
    Build a ``Progress`` and, when a UI is available, echo notes to its log so
    the on-screen log and the task row agree.
    """
    if player is not None and hasattr(player, "write_log"):
        return _UiProgress(task_id, tool, player)
    return Progress(task_id, tool)


__all__ = ["Progress", "make"]
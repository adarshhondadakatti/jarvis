"""
Headless check that the TASKS panel builds and refreshes.

    QT_QPA_PLATFORM=offscreen python tests/manual_tasks_ui_check.py

Boots the real MainWindow without a display, seeds a few tasks, clicks the
button, and asserts the pane shows them. Exits non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Redirect the DB before anything imports it.
tmp = Path(tempfile.mkdtemp(prefix="jarvis_ui_"))
import memory.db as db

db.DB_PATH = tmp / "jarvis_memory.db"
db.MEDIA_DIR = tmp / "media"
db.init_db()

from memory import task_store

# Seed: one running, one done, one failed.
a = task_store.create("web_search", {"query": "gemini live api"})
task_store.mark_running(a.id)
task_store.update(a.id, progress=0.4, progress_note="fetching results")

b = task_store.create("file_processor", {"file_path": "report.pdf"})
task_store.mark_running(b.id)
task_store.update(b.id, status=task_store.DONE, result="3 page summary")

c = task_store.create("dev_agent", {"description": "build a CLI"})
task_store.mark_running(c.id)
task_store.update(c.id, status=task_store.FAILED, error="ModuleNotFoundError: requests")

FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = ""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


from PyQt6.QtWidgets import QApplication
import ui as ui_mod

app = QApplication.instance() or QApplication(sys.argv)

# The module builds a face icon path; it tolerates a missing file, but give it one.
face = tmp / "face.png"
face.write_bytes(b"\x89PNG\r\n\x1a\n")

win = ui_mod.MainWindow(str(face))
win.show()
app.processEvents()

check("window builds", win is not None)
check("TASKS button exists", hasattr(win, "_tasks_btn"))
check("tasks timer exists", hasattr(win, "_tasks_timer"))
check("task view starts off", win._tasks_mode is False)

# Click the button — as a user would.
win._toggle_tasks()
app.processEvents()

check("task mode on after click", win._tasks_mode is True)
check("button shows active marker", "●" in win._tasks_btn.text(), win._tasks_btn.text())

title = win._content_title_lbl.text()
body = win._content_display.toPlainText()

check("pane titled for tasks", "TASK" in title.upper(), title)
check("pane lists running tool", "web_search" in body, body[:200])
check("pane lists finished tool", "file_processor" in body, body[:200])
check("pane shows failure", "dev_agent" in body and "requests" in body, body[:200])
check("pane shows percentage", "40%" in body, body[:200])
check("pane is visible", win._content_panel.isVisible())

# A second click turns it off and stops the timer.
win._toggle_tasks()
check("task mode off after second click", win._tasks_mode is False)
check("timer stopped", not win._tasks_timer.isActive())
check("button label reset", "●" not in win._tasks_btn.text(), win._tasks_btn.text())

# News taking the pane must end task mode too.
win._toggle_tasks()
win._show_content("NEWS — headlines", "some news text")
check("news ends task mode", win._tasks_mode is False)
check("news replaced pane text", "some news text" in win._content_display.toPlainText())

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("Task panel checks passed.")
print()
print("--- what the pane renders ---")
print(body)
sys.exit(0)
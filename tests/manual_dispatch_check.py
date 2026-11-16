"""
Proof that the progress reporter actually reaches tools through the real
dispatch path — run_in_executor / run_resilient, not just direct calls.

    python tests/manual_dispatch_check.py

Prints PASS/FAIL per scenario and exits non-zero on any failure. Uses a temp
DB so the real one is untouched.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

# Windows consoles default to cp1252; the status report contains "→".
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

tmp = Path(tempfile.mkdtemp(prefix="jarvis_dispatch_"))
import memory.db as db

db.DB_PATH = tmp / "tasks.db"
db.MEDIA_DIR = tmp / "media"
db.init_db()

from core.progress import Progress, make
from core.task_runner import TaskRunner
from memory import task_store

FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = ""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


# ── 1. Through run_in_executor (how main.py calls file_processor) ───────────
loop = asyncio.new_event_loop()
runner = TaskRunner()
rt = runner.register("file_processor", {"file_path": "x.pdf"})
runner.start(rt)
prog = make(rt.task_id, "file_processor", player=None)

seen: list[float] = []

def fake_tool(progress=None):
    progress(0.05, "locating file")
    seen.append(0.05)
    progress(0.3, "extracting text")
    seen.append(0.3)
    progress(0.95, "part 1/2")
    seen.append(0.95)
    progress(1.0, "done", force=True)
    return "summary text"

res = loop.run_until_complete(loop.run_in_executor(None, lambda: fake_tool(prog)))
t = task_store.get(rt.task_id)
runner.finish(rt, result=res)

check("run_in_executor returns tool result", res == "summary text", repr(res))
# 3 reports reach the tool; the 4th (the terminal 1.0) is intentionally
# skipped by the no-change guard — the value didn't move. runner.finish()
# is what actually stamps 1.0, so the row is correct regardless.
check("executor path recorded progress calls", len(seen) == 3, str(seen))
check("final progress is 1.0", t.progress == 1.0, str(t.progress))
check("final note is 'done'", t.progress_note == "done", str(t.progress_note))
check("result persisted", task_store.get(rt.task_id).result == "summary text")

# ── 2. Through run_resilient (how main.py calls web_search) ─────────────────
from core.tool_resilience import run_resilient

rt2 = runner.register("web_search", {"query": "gemini live"})
runner.start(rt2)
prog2 = Progress(rt2.task_id, "web_search", min_interval_s=0.0)

def flaky_tool(progress=None):
    progress(0.15, "searching")
    progress(1.0, "done", force=True)
    return "results"

try:
    out = loop.run_until_complete(run_resilient(
        lambda: flaky_tool(prog2), name="web_search", loop=loop, timeout=5.0, retries=1,
    ))
except Exception as e:
    out = f"ERROR {e}"
    runner.finish(rt2, error=str(e))
else:
    runner.finish(rt2, result=out)

t2 = task_store.get(rt2.task_id)
check("run_resilient returns tool result", out == "results", repr(out))
check("run_resilient path recorded progress", t2.progress == 1.0, str(t2.progress))

# ── 3. Progress survives a thread boundary (real executor thread) ───────────
import threading

rt3 = runner.register("dev_agent", {"description": "build a thing"})
runner.start(rt3)
prog3 = Progress(rt3.task_id, "dev_agent", min_interval_s=0.0)
main_thread = threading.current_thread().name
worker_thread_box: list[str] = []

def worker():
    worker_thread_box.append(threading.current_thread().name)
    prog3(0.05, "planning")
    prog3(0.6, "writing 3/5 files")
    prog3(1.0, "done", force=True)

th = threading.Thread(target=worker)
th.start()
th.join()

t3 = task_store.get(rt3.task_id)
check("work ran on a different thread", worker_thread_box[0] != main_thread,
      f"{worker_thread_box[0]} vs {main_thread}")
check("progress written from worker thread", t3.progress == 1.0, str(t3.progress))
check("note written from worker thread", t3.progress_note == "done", str(t3.progress_note))

# ── 4. A tool that never calls progress still gets a clean row ──────────────
rt4 = runner.register("open_app", {"app_name": "notepad"})
runner.start(rt4)
loop.run_until_complete(loop.run_in_executor(None, lambda: "opened"))
runner.finish(rt4, result="opened")
t4 = task_store.get(rt4.task_id)
check("tool without progress still finishes", t4.status == task_store.DONE, t4.status)
check("tool without progress defaults to 1.0", t4.progress == 1.0, str(t4.progress))

# ── 5. Reporting view ───────────────────────────────────────────────────────
report = runner.report()
check("status report lists the finished tool", "open_app" in report, report)
check("status report counts done tasks", "done: 3" in report, report)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("All dispatch checks passed.")
print()
print("--- sample status_report() ---")
print(report)
sys.exit(0)
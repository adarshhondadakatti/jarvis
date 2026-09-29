"""
file_summarizer.py — JARVIS unified summarization pipeline

Replaces the per-type, hard-truncated summarization that used to live
inline in file_processor.py. Previously every file type silently cut
text at a different arbitrary limit before summarizing:
  - PDF:  text[:50000]
  - text/docx: content[:40000]
  - pptx: text[:30000]
A long document just lost everything past the cutoff, with no signal
to the user that anything was dropped.

This module splits the job into two stages used by every file type:
  1. extract_text(path)    — full extraction, no truncation
  2. summarize_text(text)  — chunked map-reduce for long documents,
                              single-shot for short ones

Results are cached by (file hash, summary style), so re-summarizing an
unchanged file is instant and skips the Gemini call entirely.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

_CACHE_DB = Path(__file__).resolve().parent.parent / "memory" / "summary_cache.db"

# Dedicated debug log, written directly by Python with an explicit flush on
# every entry — independent of terminal redirection, console buffering, or
# how the process gets shut down. Open this file in Notepad any time; no
# PowerShell commands needed.
_DEBUG_LOG = Path(__file__).resolve().parent.parent / "jarvis_debug.log"


def _log(msg: str):
    """Write one line to jarvis_debug.log immediately (also prints to console)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass


def _log_exception(context: str):
    """Write the current exception's full traceback to jarvis_debug.log immediately."""
    import traceback
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    tb = traceback.format_exc()
    print(tb)
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{ts}] {context}\n")
            f.write(tb)
            f.write("\n")
            f.flush()
    except Exception:
        pass


_CHUNK_CHARS = 9000        # target size per chunk sent to Gemini
_CHUNK_THRESHOLD = 12000   # documents shorter than this are summarized in one shot


class ScannedDocumentError(RuntimeError):
    """
    Raised when a PDF has no extractable text layer (scanned/image-based).
    Deliberately NOT caught inside summarize() — it propagates so the
    caller (file_processor.py) can fall back to image-based summarization
    instead of the fallback silently never firing.
    """
    pass


# ── Gemini client ────────────────────────────────────────────────────────

def _get_api_key() -> str:
    config_path = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _gemini_client():
    from google import genai
    _c = genai.Client(api_key=_get_api_key())

    class _W:
        def generate_content(self, contents):
            return _c.models.generate_content(model="gemini-3.6-flash", contents=contents)

    return _W()


# ── Cache ────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    _CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CACHE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            file_hash  TEXT NOT NULL,
            style      TEXT NOT NULL,
            summary    TEXT NOT NULL,
            created_at REAL,
            PRIMARY KEY (file_hash, style)
        )
    """)
    return conn


def _file_hash(path: Path) -> str:
    """Hash of path + size + mtime — cheap fingerprint, no need to read the file twice."""
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime}"
    return hashlib.sha256(key.encode()).hexdigest()


def _cache_get(file_hash: str, style: str) -> str | None:
    conn = _connect()
    row = conn.execute(
        "SELECT summary FROM summaries WHERE file_hash = ? AND style = ?",
        (file_hash, style)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _cache_set(file_hash: str, style: str, summary: str):
    conn = _connect()
    conn.execute(
        "INSERT OR REPLACE INTO summaries (file_hash, style, summary, created_at) VALUES (?, ?, ?, ?)",
        (file_hash, style, summary, time.time())
    )
    conn.commit()
    conn.close()


# ── Stage 1: extraction (no truncation) ─────────────────────────────────

def extract_text(path: Path) -> str:
    """
    Full text extraction. No character limit — chunking for the LLM call
    happens separately in summarize_text(). Raises RuntimeError with a
    clear message when extraction isn't possible (e.g. scanned PDF with
    no text layer — caller should fall back to image-based summarization).
    """
    ext = path.suffix.lower().lstrip(".")

    if ext == "pdf":
        return _extract_pdf(path)
    if ext in ("docx", "doc"):
        return _extract_docx(path)
    if ext in ("txt", "md", "rst", "log"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext in ("pptx", "ppt"):
        return _extract_pptx(path)

    raise RuntimeError(f"No text extractor available for .{ext} files.")


def _extract_pdf(path: Path) -> str:
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        if text.strip():
            _log(f"extracted via pdfplumber ({len(text)} chars)")
            return text
    except ImportError:
        pass
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        if text.strip():
            _log(f"extracted via PyPDF2 ({len(text)} chars)")
            return text
    except ImportError:
        pass
    try:
        import fitz
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        if text.strip():
            _log(f"extracted via fitz/PyMuPDF ({len(text)} chars)")
    except Exception as e:
        _log(f"fitz extraction failed: {e}")

    if not text.strip():
        _log(f"no text layer found in {path.name} — treating as scanned/image-based")
        raise ScannedDocumentError(
            f"'{path.name}' appears to be a scanned/image-based PDF with no text layer."
        )
    return text


def _extract_docx(path: Path) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_pptx(path: Path) -> str:
    from pptx import Presentation
    prs = Presentation(path)
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        slide_text = f"\n--- Slide {i} ---\n"
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text += shape.text.strip() + "\n"
        parts.append(slide_text)
    return "\n".join(parts)


# ── Stage 2: chunked summarization ──────────────────────────────────────

def _chunk(text: str, size: int = _CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries where possible, to avoid cutting mid-sentence."""
    chunks, buf = [], ""
    for para in text.split("\n\n"):
        if len(buf) + len(para) > size and buf:
            chunks.append(buf)
            buf = para
        else:
            buf += ("\n\n" if buf else "") + para
    if buf:
        chunks.append(buf)
    return chunks


_STYLE_PROMPTS = {
    "concise":  "Summarize this document concisely, in a few clear paragraphs.",
    "bullets":  "Summarize this document as a clear, well-organized bullet list.",
    "detailed": "Give a thorough, detailed summary of this document, covering all major points.",
}


def _generate_with_retry(model, prompt: str, max_retries: int = 3, base_delay: float = 4.0):
    """
    Calls model.generate_content() with retry-with-backoff for transient
    server-side failures (503 UNAVAILABLE / "high demand", 429 rate limits).
    Does NOT retry on 404 (deprecated model, permanent) or other client
    errors that a retry can't fix.
    """
    from google.genai.errors import ServerError, ClientError
    import time as _time

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return model.generate_content(prompt)
        except ServerError as e:
            last_error = e
            _log(f"Gemini server error (attempt {attempt}/{max_retries}): {e}")
        except ClientError as e:
            # 429 (rate limit) is worth retrying; anything else (404, bad
            # request, etc.) is permanent — fail immediately.
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                last_error = e
                _log(f"Gemini rate limit (attempt {attempt}/{max_retries}): {e}")
            else:
                raise

        if attempt < max_retries:
            delay = base_delay * attempt  # 4s, 8s, 12s...
            _log(f"Retrying in {delay:.0f}s...")
            _time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini summarization failed: max retries reached without response.")


def summarize_text(text: str, style: str = "concise") -> str:
    """
    Map-reduce summarization: short documents get one Gemini call; long
    documents are chunked, each chunk summarized, then the partial
    summaries are combined into a single coherent final summary — instead
    of silently dropping everything past a fixed character limit.
    Transient server errors (503 high-demand, 429 rate limits) are retried
    automatically before giving up.
    """
    prompt_instruction = _STYLE_PROMPTS.get(style, _STYLE_PROMPTS["concise"])
    model = _gemini_client()

    if len(text) <= _CHUNK_THRESHOLD:
        response = _generate_with_retry(model, f"{prompt_instruction}\n\n{text}")
        return response.text.strip()

    chunks = _chunk(text)
    partial_summaries = []
    for i, chunk in enumerate(chunks, 1):
        resp = _generate_with_retry(
            model,
            f"This is part {i} of {len(chunks)} of a longer document. "
            f"Summarize just this part concisely, preserving key facts and figures:\n\n{chunk}"
        )
        partial_summaries.append(resp.text.strip())

    combined = "\n\n".join(partial_summaries)
    final = _generate_with_retry(
        model,
        f"{prompt_instruction} The following are summaries of consecutive parts "
        f"of the same document — combine them into one coherent overall summary, "
        f"removing redundancy:\n\n{combined}"
    )
    return final.text.strip()


# ── Orchestration ────────────────────────────────────────────────────────

def summarize_file(path: Path, style: str = "concise", force: bool = False) -> dict:
    """
    Full pipeline: cache check → extract → summarize → cache store.
    Returns {"summary": str, "cached": bool, "chars": int}.
    """
    file_hash = _file_hash(path)

    if not force:
        cached = _cache_get(file_hash, style)
        if cached:
            return {"summary": cached, "cached": True, "chars": len(cached)}

    text = extract_text(path)
    if not text.strip():
        raise RuntimeError("Document appears to be empty.")

    summary = summarize_text(text, style=style)
    _cache_set(file_hash, style, summary)
    return {"summary": summary, "cached": False, "chars": len(text)}


# ── Tool-facing wrapper — call this from file_processor.py's dispatch   ──
# ── for the "summarize" action across pdf / docx / text / pptx types.   ──

def summarize(parameters: dict = None, player=None) -> str:
    params = parameters or {}
    file_path_str = params.get("file_path", "").strip()
    if not file_path_str:
        _log("summarize() called with no file_path")
        return "No file path provided."

    _log(f"summarize() requested: '{file_path_str}'")

    path = Path(file_path_str)
    if not path.exists():
        from actions.file_finder import find_best_match
        found = find_best_match(file_path_str)
        if not found:
            _log(f"summarize(): no match found for '{file_path_str}'")
            return f"File not found: {file_path_str}"
        path = found
        _log(f"summarize(): resolved '{file_path_str}' -> {path}")

    style = params.get("style", "concise")
    force = bool(params.get("force", False))

    if player:
        player.write_log(f"[file_summarizer] {path.name} (style={style})")

    try:
        result = summarize_file(path, style=style, force=force)
    except ScannedDocumentError as e:
        _log(f"summarize(): ScannedDocumentError — {e}")
        # Deliberately NOT swallowed here — let it propagate so
        # file_processor.py's caller can fall back to image-based
        # summarization instead of this just becoming a dead-end message.
        raise
    except RuntimeError as e:
        _log(f"summarize(): RuntimeError — {e}")
        return str(e)
    except Exception as e:
        _log_exception(f"summarize(): unexpected error on {path}")
        return f"Summarization failed: {e}"

    tag = " (cached)" if result["cached"] else ""
    summary = result["summary"]
    _log(f"summarize(): success on {path.name} ({result['chars']} chars extracted, cached={result['cached']})")
    if len(summary) > 600 and params.get("save", True):
        out = path.parent / f"{path.stem}_summary.txt"
        out.write_text(summary, encoding="utf-8")
        return f"{summary[:400]}...{tag}\n\nFull summary saved: {out.name}"
    return f"{summary}{tag}"
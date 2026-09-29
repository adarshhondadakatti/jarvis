import os
import re
import shutil
import platform
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import send2trash
    _SEND2TRASH = True
except ImportError:
    _SEND2TRASH = False

_OS = platform.system()  # "Windows" | "Darwin" | "Linux"

_SAFE_ROOTS: list[Path] = [
    Path.home(),
]

def _is_safe_path(target: Path) -> bool:
    """Verilen path _SAFE_ROOTS içinde mi? Değilse işlemi reddet."""
    try:
        resolved = target.resolve()
        return any(
            resolved == root.resolve() or resolved.is_relative_to(root.resolve())
            for root in _SAFE_ROOTS
        )
    except Exception:
        return False

def _get_desktop() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_DESKTOP_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Desktop"

def _get_downloads() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_DOWNLOAD_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    # Windows: try USERPROFILE/Downloads, then check known folder via env vars
    if _OS == "Windows":
        env_path = os.environ.get("USERPROFILE") or os.environ.get("HOMEPATH")
        if env_path:
            candidate = Path(env_path) / "Downloads"
            if candidate.exists():
                return candidate
    return Path.home() / "Downloads"

def _get_documents() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_DOCUMENTS_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Documents"

def _get_pictures() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_PICTURES_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Pictures"

def _get_music() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_MUSIC_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Music"

def _get_videos() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_VIDEOS_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Videos"


def _search_dirs() -> list[Path]:
    """Common directories to search when looking up a file by name."""
    dirs = [
        _get_desktop(),
        _get_downloads(),
        _get_documents(),
        _get_pictures(),
        _get_music(),
        _get_videos(),
        Path.home(),
    ]
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for d in dirs:
        try:
            r = d.resolve()
            if r not in seen:
                seen.add(r)
                unique.append(d)
        except Exception:
            pass
    return unique

def _find_file_by_name(name: str, max_depth: int = 3, max_files: int = 5000) -> Path | None:
    """
    Search common directories for a file matching *name*.
    Searches top-level first (fast), then recursively up to *max_depth* levels.
    Returns the first match found, or None.
    """
    name_lower = name.lower()
    for search_dir in _search_dirs():
        if not search_dir.exists() or not search_dir.is_dir():
            continue
        try:
            # Fast path: exact match at top level
            exact = search_dir / name
            if exact.is_file():
                return exact

            # Top-level case-insensitive match
            for item in search_dir.iterdir():
                if item.is_file() and item.name.lower() == name_lower:
                    return item

            # Recursive search in subdirectories (depth-limited)
            files_checked = 0
            for root, dirs, files in os.walk(search_dir):
                # Calculate current depth relative to search_dir
                rel = os.path.relpath(root, str(search_dir))
                depth = 0 if rel == "." else rel.count(os.sep) + 1
                if depth >= max_depth:
                    dirs[:] = []  # Don't descend further
                    continue
                for fname in files:
                    files_checked += 1
                    if files_checked > max_files:
                        dirs[:] = []
                        break
                    if fname.lower() == name_lower:
                        return Path(root) / fname
        except (PermissionError, OSError):
            continue
    return None


#: Extensions that are documents worth summarising / reading.
_DOCUMENT_EXTS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".rst", ".log",
    ".pptx", ".ppt", ".csv", ".xlsx", ".xls", ".json", ".xml",
    ".html", ".htm", ".rtf", ".odt",
}


def _score_fuzzy_match(query: str, filename: str) -> float:
    """Score how well *filename* matches *query* (0 = no match, higher = better).

    The query is typically a keyword the user spoke — e.g. ``"marksheet"``
    for a file called ``"nandeesh_marksheet.pdf"``.  A substring check is
    the primary signal; word-boundary alignment, document extension, and
    filename brevity provide refinements so that among several hits the one
    the user most likely meant ranks first.
    """
    fn_lower = filename.lower()
    q_lower  = query.lower().strip()
    if not q_lower:
        return 0.0

    # Tokenise the query on whitespace / underscore boundaries so that
    # multi-word queries like "marksheet 2024" are handled gracefully.
    tokens = [t for t in re.split(r"[\s_]+", q_lower) if t and len(t) >= 2]
    if not tokens:
        return 0.0

    # ALL tokens must appear somewhere in the filename — allowing
    # partial matches causes false positives like "doc" inside "mkdocs".
    matched = [t for t in tokens if t in fn_lower]
    if len(matched) != len(tokens):
        return 0.0

    score = 1.0  # every token matched

    # Bonus when the *entire* query string appears as a substring — this is
    # the common case ("marksheet" inside "nandeesh_marksheet.pdf").
    if q_lower in fn_lower:
        score += 0.5

    # Bonus when a token aligns at a word boundary (preceded or followed by
    # a non-alphanumeric separator).  ``"marksheet"`` matching
    # ``_marksheet.`` in ``nandeesh_marksheet.pdf`` gets this; a query
    # ``"mark"`` matching ``"remarkable.pdf"`` does not.
    boundary_match = 0
    for t in tokens:
        pattern = r"(?:^|[^a-z0-9])" + re.escape(t) + r"(?:[^a-z0-9]|$)"
        if re.search(pattern, fn_lower):
            boundary_match += 1
    score += 0.3 * (boundary_match / len(tokens))

    # Shorter filenames are more likely the intended target.
    score += 2.0 / (2.0 + len(fn_lower))

    # Boost recognised document extensions.
    ext = Path(filename).suffix.lower()
    if ext in _DOCUMENT_EXTS:
        score += 0.1

    return round(score, 4)


def _find_file_fuzzy(
    query: str,
    max_results: int = 10,
    prefer_extensions: set[str] | None = None,
) -> list[Path]:
    """Fuzzy file lookup — finds files whose names contain *query*.

    Unlike :func:`_find_file_by_name` (which requires an exact name),
    this function performs **substring** matching so that a spoken
    description like ``"marksheet"`` resolves to
    ``"nandeesh_marksheet.pdf"``.

    Args:
        query:            A keyword or partial filename to search for.
        max_results:      Maximum number of candidates to return.
        prefer_extensions: Set of extensions (e.g. ``{".pdf", ".docx"}``)
                          to rank higher in the results.

    Returns:
        List of :class:`~pathlib.Path` objects ranked best-match-first.
        Empty list when nothing is found.
    """
    query = query.strip()
    if not query:
        return []

    candidates: list[tuple[float, Path]] = []
    seen: set[Path] = set()  # de-dup across overlapping search dirs

    for search_dir in _search_dirs():
        if not search_dir.exists() or not search_dir.is_dir():
            continue
        try:
            for root, dirs, files in os.walk(search_dir):
                # Depth limit — keep the search fast on large trees.
                rel   = os.path.relpath(root, str(search_dir))
                depth = 0 if rel == "." else rel.count(os.sep) + 1
                if depth >= 2:               # only descend 2 levels
                    dirs[:] = []
                    continue

                for fname in files:
                    score = _score_fuzzy_match(query, fname)
                    if score <= 0:
                        continue

                    full_path = Path(root) / fname
                    resolved = full_path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)

                    # Apply extension preference.
                    ext = Path(fname).suffix.lower()
                    if prefer_extensions is not None and ext in prefer_extensions:
                        score += 0.2

                    candidates.append((score, full_path))
        except (PermissionError, OSError):
            continue

    # Sort: higher score first, then shorter filename, then path string
    # for deterministic ordering.
    candidates.sort(key=lambda c: (-c[0], len(c[1].name), str(c[1])))

    return [p for _, p in candidates[:max_results]]


def _open_file_in_app(path: Path) -> str:
    """
    Open a file in its default desktop application.
    Uses os.startfile() on Windows, 'open' on macOS, 'xdg-open' on Linux.
    """
    if not path.exists():
        return f"File not found: {path.name}"
    if not path.is_file():
        return f"Not a file: {path.name}"

    try:
        if _OS == "Windows":
            os.startfile(str(path))
        elif _OS == "Darwin":
            subprocess.run(["open", str(path)], check=True)
        else:
            subprocess.run(["xdg-open", str(path)], check=True)
        return f"Opened: {path.name}"
    except FileNotFoundError:
        return f"No default application found for: {path.name}"
    except Exception as e:
        return f"Could not open {path.name}: {e}"

def _resolve_path(raw: str) -> Path:
    shortcuts: dict[str, Path] = {
        "desktop":   _get_desktop(),
        "downloads": _get_downloads(),
        "documents": _get_documents(),
        "pictures":  _get_pictures(),
        "music":     _get_music(),
        "videos":    _get_videos(),
        "home":      Path.home(),
    }
    lower = raw.strip().lower()
    if lower in shortcuts:
        return shortcuts[lower]

    candidate = Path(raw).expanduser()

    # If the path exists as-is (absolute or relative), use it directly
    if candidate.exists():
        return candidate

    # If it looks like a bare filename (no path separators), resolve it via
    # the shared, indexed fuzzy search instead of scanning the disk here.
    if "/" not in raw and "\\" not in raw and "~" not in raw:
        from actions.file_finder import find_best_match
        found = find_best_match(raw)
        if found:
            return found

    return candidate

def _format_size(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"

def _safe_trash(target: Path) -> str:

    if not _SEND2TRASH:
        return (
            "send2trash is not installed. "
            "Run: pip install send2trash — "
            "Permanent deletion is disabled for safety."
        )
    send2trash.send2trash(str(target))
    return f"Moved to Trash: {target.name}"


def list_files(path: str = "desktop", show_hidden: bool = False) -> str:
    try:
        target = _resolve_path(path)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists():
            return f"Path not found: {target}"
        if not target.is_dir():
            return f"Not a directory: {target}"

        items = []
        for item in sorted(target.iterdir()):
            if not show_hidden and item.name.startswith("."):
                continue
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            else:
                size = _format_size(item.stat().st_size)
                items.append(f"📄 {item.name} ({size})")

        if not items:
            return f"Directory is empty: {target.name}/"

        return f"Contents of {target.name}/ ({len(items)} items):\n" + "\n".join(items)

    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Error listing files: {e}"


def create_file(path: str, name: str = "", content: str = "") -> str:
    try:
        base   = _resolve_path(path)
        target = (base / name) if name else base
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"File created: {target.name}"
    except Exception as e:
        return f"Could not create file: {e}"


def create_folder(path: str, name: str = "") -> str:
    try:
        base   = _resolve_path(path)
        target = (base / name) if name else base
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        target.mkdir(parents=True, exist_ok=True)
        return f"Folder created: {target.name}"
    except Exception as e:
        return f"Could not create folder: {e}"


def delete_file(path: str, name: str = "") -> str:
    try:
        base   = _resolve_path(path)
        target = (base / name) if name else base
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists():
            return f"Not found: {target.name}"

        # Güvenli dizin kontrolü — kritik kullanıcı klasörlerini koru
        protected = {
            _get_desktop(), _get_downloads(), _get_documents(),
            _get_pictures(), _get_music(), _get_videos(), Path.home()
        }
        if target.resolve() in {p.resolve() for p in protected}:
            return f"Protected directory, cannot delete: {target.name}"

        return _safe_trash(target)

    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Could not delete: {e}"


def move_file(path: str, name: str = "", destination: str = "") -> str:
    try:
        base   = _resolve_path(path)
        src    = (base / name) if name else base
        dst    = _resolve_path(destination) if destination else None

        if not src.exists():
            return f"Source not found: {src.name}"
        if dst is None:
            return "No destination specified."
        if not _is_safe_path(src):
            return f"Access denied (source): {src}"
        if not _is_safe_path(dst):
            return f"Access denied (destination): {dst}"

        if dst.is_dir():
            dst = dst / src.name

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Moved: {src.name} → {dst.parent.name}/"

    except Exception as e:
        return f"Could not move: {e}"


def copy_file(path: str, name: str = "", destination: str = "") -> str:
    try:
        base = _resolve_path(path)
        src  = (base / name) if name else base
        dst  = _resolve_path(destination) if destination else None

        if not src.exists():
            return f"Source not found: {src.name}"
        if dst is None:
            return "No destination specified."
        if not _is_safe_path(src):
            return f"Access denied (source): {src}"
        if not _is_safe_path(dst):
            return f"Access denied (destination): {dst}"

        if dst.is_dir():
            dst = dst / src.name

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

        return f"Copied: {src.name} → {dst.parent.name}/"

    except Exception as e:
        return f"Could not copy: {e}"


def rename_file(path: str, name: str = "", new_name: str = "") -> str:
    try:
        base     = _resolve_path(path)
        target   = (base / name) if name else base
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists():
            return f"Not found: {target.name}"
        if not new_name:
            return "No new name provided."

        new_path = target.parent / new_name
        if new_path.exists():
            return f"A file named '{new_name}' already exists here."

        target.rename(new_path)
        return f"Renamed: {target.name} → {new_name}"

    except Exception as e:
        return f"Could not rename: {e}"


def read_file(path: str, name: str = "", max_chars: int = 4000) -> str:
    try:
        base   = _resolve_path(path)
        target = (base / name) if name else base
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists():
            return f"File not found: {target.name}"
        if not target.is_file():
            return f"Not a file: {target.name}"

        content = target.read_text(encoding="utf-8", errors="ignore")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[Truncated — {len(content)} total chars]"
        return content

    except Exception as e:
        return f"Could not read file: {e}"


def write_file(path: str, name: str = "", content: str = "",
               append: bool = False) -> str:
    try:
        base   = _resolve_path(path)
        target = (base / name) if name else base
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(target, mode, encoding="utf-8") as f:
            f.write(content)
        action = "Appended to" if append else "Written to"
        return f"{action}: {target.name}"
    except Exception as e:
        return f"Could not write file: {e}"


def find_files(name: str = "", extension: str = "",
               path: str = "home", max_results: int = 20) -> str:
    """
    Delegates to the shared, indexed search in file_finder.py instead of
    doing its own disk walk. `path` is accepted for backward compatibility
    with existing callers but no longer scopes the search — the index
    already covers all default roots (Desktop, Downloads, Documents,
    Pictures, Music, Videos, home) with noisy directories excluded and
    fuzzy ranking applied.
    """
    from actions.file_finder import find_files as _indexed_find
    return _indexed_find(name=name, extension=extension, max_results=min(max_results, 50))


def get_largest_files(path: str = "downloads", count: int = 10) -> str:
    count = min(count, 50)  # maksimum 50
    try:
        search_path = _resolve_path(path)
        if not _is_safe_path(search_path):
            return f"Access denied: {search_path}"
        if not search_path.exists():
            return f"Path not found: {path}"

        files = []
        for item in search_path.rglob("*"):
            if item.is_file():
                try:
                    files.append((item.stat().st_size, item))
                except Exception:
                    continue

        files.sort(reverse=True)
        top = files[:count]

        if not top:
            return "No files found."

        lines = [f"Top {len(top)} largest files in {search_path.name}/:"]
        for size, f in top:
            lines.append(f"  {_format_size(size):>10}  {f.name}  ({f.parent})")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: {e}"


def get_disk_usage(path: str = "home") -> str:
    try:
        target = _resolve_path(path)
        usage  = shutil.disk_usage(target)
        pct    = usage.used / usage.total * 100
        return (
            f"Disk usage ({target}):\n"
            f"  Total : {_format_size(usage.total)}\n"
            f"  Used  : {_format_size(usage.used)} ({pct:.1f}%)\n"
            f"  Free  : {_format_size(usage.free)}"
        )
    except Exception as e:
        return f"Could not get disk usage: {e}"


def organize_desktop() -> str:
    type_map = {
        "Images":    {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".ico", ".heic"},
        "Documents": {".pdf", ".doc", ".docx", ".txt", ".xls", ".xlsx",
                      ".ppt", ".pptx", ".csv", ".odt", ".ods", ".odp"},
        "Videos":    {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"},
        "Music":     {".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"},
        "Archives":  {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
        "Code":      {".py", ".js", ".ts", ".html", ".css", ".json", ".xml",
                      ".cpp", ".java", ".cs", ".go", ".rs", ".sh"},
    }

    desktop = _get_desktop()
    moved, skipped = [], []

    try:
        for item in desktop.iterdir():
            # Klasörlere, gizli dosyalara ve organize klasörlerine dokunma
            if item.is_dir() or item.name.startswith("."):
                continue
            if item.name in {k for k in type_map}:
                continue

            ext        = item.suffix.lower()
            target_dir = desktop / "Others"
            for folder, exts in type_map.items():
                if ext in exts:
                    target_dir = desktop / folder
                    break

            target_dir.mkdir(exist_ok=True)
            new_path = target_dir / item.name

            if new_path.exists():
                skipped.append(item.name)
                continue

            shutil.move(str(item), str(new_path))
            moved.append(f"{item.name} → {target_dir.name}/")

        result = f"Desktop organized: {len(moved)} files moved."
        if moved:
            preview = moved[:8]
            result += "\n" + "\n".join(preview)
            if len(moved) > 8:
                result += f"\n... and {len(moved) - 8} more."
        if skipped:
            result += f"\n{len(skipped)} file(s) skipped (name conflict)."
        return result

    except Exception as e:
        return f"Could not organize desktop: {e}"


def get_file_info(path: str, name: str = "") -> str:
    try:
        base   = _resolve_path(path)
        target = (base / name) if name else base
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists():
            return f"Not found: {target.name}"

        stat = target.stat()
        info = {
            "Name":      target.name,
            "Type":      "Folder" if target.is_dir() else "File",
            "Size":      _format_size(stat.st_size),
            "Location":  str(target.parent),
            "Created":   datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M"),
            "Modified":  datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "Extension": target.suffix or "—",
        }
        return "\n".join(f"  {k}: {v}" for k, v in info.items())

    except Exception as e:
        return f"Could not get file info: {e}"

def file_controller(
    parameters: dict = None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    action = params.get("action", "").lower().strip()
    path   = params.get("path", "desktop")
    name   = params.get("name", "")

    if player:
        player.write_log(f"[file] {action} {name or path}")

    try:
        if action == "list":
            return list_files(path)

        elif action == "create_file":
            return create_file(path, name=name, content=params.get("content", ""))

        elif action == "create_folder":
            return create_folder(path, name=name)

        elif action == "delete":
            return delete_file(path, name=name)

        elif action == "move":
            return move_file(path, name=name, destination=params.get("destination", ""))

        elif action == "copy":
            return copy_file(path, name=name, destination=params.get("destination", ""))

        elif action == "rename":
            return rename_file(path, name=name, new_name=params.get("new_name", ""))

        elif action == "read":
            return read_file(path, name=name)

        elif action == "open":
            base   = _resolve_path(path)
            target = (base / name) if name else base
            if not _is_safe_path(target):
                return f"Access denied: {target}"
            return _open_file_in_app(target)

        elif action == "write":
            return write_file(
                path, name=name,
                content=params.get("content", ""),
                append=params.get("append", False)
            )

        elif action == "find":
            # Kept for backward compatibility — new code should call the
            # standalone file_finder tool directly instead of routing
            # through file_controller.
            return find_files(
                name=name or params.get("name", ""),
                extension=params.get("extension", ""),
                path=path,
                max_results=min(int(params.get("max_results", 20)), 50),
            )

        elif action == "largest":
            return get_largest_files(
                path=path,
                count=int(params.get("count", 10)),
            )

        elif action == "disk_usage":
            return get_disk_usage(path)

        elif action == "organize_desktop":
            return organize_desktop()

        elif action == "info":
            return get_file_info(path, name=name)

        else:
            return f"Unknown action: '{action}'"

    except Exception as e:
        return f"File controller error ({action}): {e}"
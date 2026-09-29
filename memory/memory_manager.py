"""
JARVIS Memory Manager — OKF Edition

Text memory is stored in Open Knowledge Format (OKF): a directory of
markdown files with YAML frontmatter. Each memory entry becomes a
"concept" file, organized by category.

Structure:
    memory/okf/<profile>/
        index.md          # Catalog of all concepts
        log.md            # Chronological record
        identity/
            name.md       # concept: type=identity
        preferences/
            theme.md
        ...

This module no longer uses SQLite for text memory. The database
(memory/db.py) is still used by face_memory, event_memory, media_memory,
and vector_store.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR         = get_base_dir()
OKF_BASE_DIR     = BASE_DIR / "memory" / "okf"

# ── Constants ──────────────────────────────────────────────────────────────────

VALID_CATEGORIES = ("identity", "preferences", "projects", "relationships", "wishes", "notes")
MAX_VALUE_LENGTH = 380
MEMORY_MAX_CHARS = 2200
MEMORY_VERSION   = "1.0"
DEFAULT_PROFILE  = "default"

# Category → OKF type mapping
CATEGORY_TO_TYPE = {
    "identity":       "identity",
    "preferences":    "preference",
    "projects":       "project",
    "relationships":  "relationship",
    "wishes":         "wish",
    "notes":          "note",
}

TYPE_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_TYPE.items()}

# ── Profile management ─────────────────────────────────────────────────────────

_current_profile = DEFAULT_PROFILE


def get_profile() -> str:
    """Return the currently active memory profile."""
    return _current_profile


def set_profile(profile: str) -> str:
    """
    Set the active memory profile. Profiles are stored as subdirectories
    under memory/okf/. Returns the previous profile name.
    """
    global _current_profile
    old = _current_profile
    _current_profile = profile.strip() or DEFAULT_PROFILE
    print(f"[Memory] Profile switched: {old} -> {_current_profile}")
    return old


def get_profiles() -> list[str]:
    """Return a list of all profile names (subdirectories under okf/)."""
    if not OKF_BASE_DIR.exists():
        return []
    return sorted(
        d.name for d in OKF_BASE_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def list_profiles() -> str:
    """List all available memory profiles."""
    profiles = get_profiles()
    if not profiles:
        return "No profiles found. Memory is empty."
    current = get_profile()
    lines = ["Available memory profiles:"]
    for p in profiles:
        marker = " (current)" if p == current else ""
        lines.append(f"  - {p}{marker}")
    return "\n".join(lines)


# ── Initialization ─────────────────────────────────────────────────────────────

def init_memory() -> None:
    """
    Initialise the OKF memory store.
    Creates the OKF directory structure if it doesn't exist.
    """
    OKF_BASE_DIR.mkdir(parents=True, exist_ok=True)
    # Create default profile directory
    (OKF_BASE_DIR / DEFAULT_PROFILE).mkdir(exist_ok=True)


def _profile_dir(profile: str | None = None) -> Path:
    """Return the OKF directory for a profile."""
    prof = profile or _current_profile
    d = OKF_BASE_DIR / prof
    d.mkdir(parents=True, exist_ok=True)
    return d


def _empty_memory() -> dict:
    return {
        "identity":      {},
        "preferences":   {},
        "projects":      {},
        "relationships": {},
        "wishes":        {},
        "notes":         {},
    }


# ── OKF file I/O ───────────────────────────────────────────────────────────────

def _concept_path(profile: str, category: str, key: str) -> Path:
    """Return the file path for a concept file."""
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    if not safe_key:
        safe_key = "entry"
    return _profile_dir(profile) / category / f"{safe_key}.md"


def _write_concept(profile: str, category: str, key: str, value: str,
                   updated: str | None = None) -> None:
    """Write a single concept file in OKF format."""
    if updated is None:
        updated = datetime.now().strftime("%Y-%m-%d")

    file_path = _concept_path(profile, category, key)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    frontmatter = {
        "type": CATEGORY_TO_TYPE.get(category, "note"),
        "title": key.replace("_", " ").title(),
        "description": value[:200] if len(value) > 200 else value,
        "tags": [category, key],
        "sources": [
            {
                "generated": "jarvis-memory",
                "last_modified": updated,
            }
        ],
        "status": "stable",
    }

    try:
        import yaml
        yaml_text = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    except ImportError:
        # Fallback: manual YAML
        yaml_text = _manual_yaml(frontmatter)

    file_path.write_text(
        f"---\n{yaml_text}---\n\n{value}\n",
        encoding="utf-8",
    )


def _manual_yaml(data: dict) -> str:
    """Simple YAML serializer fallback (no external dependency)."""
    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"  -")
                    for k, v in item.items():
                        lines.append(f"    {k}: {json.dumps(v) if isinstance(v, str) else v}")
                else:
                    lines.append(f"  - {json.dumps(item) if isinstance(item, str) else item}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                lines.append(f"  {k}: {json.dumps(v) if isinstance(v, str) else v}")
        else:
            lines.append(f"{key}: {json.dumps(value) if isinstance(value, str) else value}")
    return "\n".join(lines) + "\n"


def _read_concept(file_path: Path) -> tuple[str, str, str] | None:
    """
    Read a concept file. Returns (category, key, value) or None.
    Category is inferred from the directory name.
    Key is inferred from the frontmatter title or filename.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return None

        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter_text = parts[1]
        body = parts[2].strip()

        try:
            import yaml
            fm = yaml.safe_load(frontmatter_text)
        except ImportError:
            fm = _parse_simple_yaml(frontmatter_text)

        if not isinstance(fm, dict):
            return None

        # Category from directory name
        category = file_path.parent.name
        if category not in VALID_CATEGORIES:
            # Try to infer from type
            concept_type = fm.get("type", "")
            category = TYPE_TO_CATEGORY.get(concept_type, "notes")

        # Key from title or filename
        key = fm.get("title", file_path.stem).lower().replace(" ", "_")

        return (category, key, body)
    except Exception:
        return None


def _parse_simple_yaml(text: str) -> dict:
    """Simple YAML parser fallback for basic frontmatter."""
    result = {}
    current_key = None
    for line in text.strip().split("\n"):
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(" "):
            if ":" in line:
                key, _, val = line.partition(":")
                current_key = key.strip()
                val = val.strip()
                if val:
                    result[current_key] = val.strip("'\"")
                else:
                    result[current_key] = {}
        elif current_key and isinstance(result.get(current_key), dict):
            if ":" in line:
                k, _, v = line.strip().partition(":")
                result[current_key][k.strip()] = v.strip().strip("'\"")
    return result


# ── Core memory operations ─────────────────────────────────────────────────────

def load_memory(profile: str | None = None) -> dict:
    """Load all memory from OKF directory for the given profile."""
    prof = profile or _current_profile
    profile_dir = _profile_dir(prof)

    memory = _empty_memory()
    for category in VALID_CATEGORIES:
        cat_dir = profile_dir / category
        if not cat_dir.exists():
            continue
        for md_file in cat_dir.glob("*.md"):
            result = _read_concept(md_file)
            if result:
                cat, key, value = result
                memory[cat][key] = {
                    "value": value,
                    "updated": datetime.now().strftime("%Y-%m-%d"),
                }

    return memory


def save_memory(memory: dict, profile: str | None = None) -> None:
    """Save all memory to OKF directory for the given profile."""
    if not isinstance(memory, dict):
        return
    prof = profile or _current_profile
    profile_dir = _profile_dir(prof)

    # Clear existing concept files
    for category in VALID_CATEGORIES:
        cat_dir = profile_dir / category
        if cat_dir.exists():
            for md_file in cat_dir.glob("*.md"):
                md_file.unlink()

    # Write new concept files
    count = 0
    for cat, items in memory.items():
        if cat not in VALID_CATEGORIES:
            continue
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            if isinstance(entry, dict) and "value" in entry:
                value = str(entry["value"])
                updated = entry.get("updated", datetime.now().strftime("%Y-%m-%d"))
                _write_concept(prof, cat, key, value, updated)
                count += 1

    # Regenerate index and log
    _write_index(prof)
    _write_log(prof)

    print(f"[Memory] Saved {count} entries to OKF (profile: {prof})")


def update_memory(memory_update: dict, profile: str | None = None) -> dict:
    """Update specific memory entries. Loads, merges, saves."""
    prof = profile or _current_profile
    if not isinstance(memory_update, dict) or not memory_update:
        return load_memory(prof)

    memory = load_memory(prof)
    changed = False

    for cat, items in memory_update.items():
        if cat not in VALID_CATEGORIES:
            continue
        if not isinstance(items, dict):
            continue
        for key, value in items.items():
            if value is None:
                continue
            if isinstance(value, dict) and "value" in value:
                new_val = str(value["value"])
                if len(new_val) > MAX_VALUE_LENGTH:
                    new_val = new_val[:MAX_VALUE_LENGTH].rstrip() + "…"
                updated = value.get("updated", datetime.now().strftime("%Y-%m-%d"))
                memory[cat][key] = {"value": new_val, "updated": updated}
                _write_concept(prof, cat, key, new_val, updated)
                changed = True
            elif isinstance(value, str):
                new_val = value[:MAX_VALUE_LENGTH] if len(value) > MAX_VALUE_LENGTH else value
                updated = datetime.now().strftime("%Y-%m-%d")
                memory[cat][key] = {"value": new_val, "updated": updated}
                _write_concept(prof, cat, key, new_val, updated)
                changed = True

    if changed:
        _write_index(prof)
        _write_log(prof)
        print(f"[Memory] Updated: {list(memory_update.keys())} (profile: {prof})")

    return memory


def forget(key: str, category: str = "notes", profile: str | None = None) -> str:
    """Remove a specific memory entry."""
    if category not in VALID_CATEGORIES:
        category = "notes"
    prof = profile or _current_profile

    file_path = _concept_path(prof, category, key)
    if file_path.exists():
        file_path.unlink()
        _write_index(prof)
        _write_log(prof)
        return f"Forgotten: {category}/{key} (profile: {prof})"
    return f"Not found: {category}/{key} (profile: {prof})"


def remember(key: str, value: str, category: str = "notes",
             profile: str | None = None) -> str:
    """Remember a new fact."""
    if category not in VALID_CATEGORIES:
        category = "notes"
    prof = profile or _current_profile
    update_memory({category: {key: {"value": value}}}, prof)
    return f"Remembered: {category}/{key} = {value} (profile: {prof})"


def switch_profile(profile: str) -> str:
    """Switch to a different memory profile."""
    if not profile or not profile.strip():
        return "Profile name cannot be empty."
    old = set_profile(profile)
    init_memory()  # Ensure the new profile directory exists
    return f"Switched memory profile: {old} → {profile}"


# ── OKF index and log ──────────────────────────────────────────────────────────

def _write_index(profile: str) -> None:
    """Write index.md (catalog of all concepts)."""
    memory = load_memory(profile)
    profile_dir = _profile_dir(profile)

    lines = [
        "# Knowledge Index",
        "",
        f"Profile: {profile}",
        f"Version: {MEMORY_VERSION}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Concepts",
        "",
    ]

    total = 0
    for cat in VALID_CATEGORIES:
        items = memory.get(cat, {})
        if not items:
            continue
        lines.append(f"### {cat.title()}")
        lines.append("")
        for key, entry in items.items():
            value = entry.get("value", "") if isinstance(entry, dict) else str(entry)
            desc = value[:100] + "..." if len(value) > 100 else value
            lines.append(f"- **{key}** — {desc}")
            total += 1
        lines.append("")

    lines.insert(3, f"Total concepts: {total}")

    (profile_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _write_log(profile: str) -> None:
    """Write log.md (chronological record)."""
    memory = load_memory(profile)
    profile_dir = _profile_dir(profile)

    lines = [
        "# Memory Log",
        "",
        f"Profile: {profile}",
        "",
    ]

    entries = []
    for cat in VALID_CATEGORIES:
        items = memory.get(cat, {})
        for key, entry in items.items():
            updated = entry.get("updated", "") if isinstance(entry, dict) else ""
            entries.append((updated, cat, key))

    entries.sort(reverse=True)
    for updated, cat, key in entries:
        lines.append(f"- {updated} — {key} ({cat})")

    (profile_dir / "log.md").write_text("\n".join(lines), encoding="utf-8")


# ── OKF export/import (now the primary storage format) ──────────────────────────

def export_okf(profile: str | None = None) -> str:
    """
    Export memory to OKF format. Since memory is already stored in OKF,
    this creates a portable copy that can be shared or backed up.
    """
    prof = profile or _current_profile
    src_dir = _profile_dir(prof)
    export_dir = OKF_BASE_DIR / f"export_{prof}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    import shutil
    shutil.copytree(src_dir, export_dir)

    return f"Memory exported to OKF at: {export_dir.name}/ (profile: {prof})"


def import_okf(file_path: str, profile: str | None = None,
               overwrite: bool = False) -> str:
    """
    Import memory from an OKF directory.
    """
    prof = profile or _current_profile
    src = Path(file_path)

    if not src.exists() or not src.is_dir():
        return f"Directory not found: {file_path}"

    if overwrite:
        # Clear existing profile directory
        profile_dir = _profile_dir(prof)
        for item in profile_dir.iterdir():
            if item.is_dir():
                import shutil
                shutil.rmtree(item)
            else:
                item.unlink()

    # Copy concept files from source
    count = 0
    for category in VALID_CATEGORIES:
        src_cat = src / category
        if not src_cat.exists():
            continue
        dst_cat = _profile_dir(prof) / category
        dst_cat.mkdir(parents=True, exist_ok=True)

        for md_file in src_cat.glob("*.md"):
            dst_file = dst_cat / md_file.name
            if not overwrite and dst_file.exists():
                continue
            import shutil
            shutil.copy2(md_file, dst_file)
            count += 1

    # Regenerate index and log
    _write_index(prof)
    _write_log(prof)

    return f"Imported {count} entries from OKF into profile '{prof}'"


# ── Backward compatibility aliases ─────────────────────────────────────────────

# These aliases maintain compatibility with code that uses the old names.
# Since memory is now stored in OKF format, these are the same as the OKF versions.
export_memory = export_okf
import_memory = import_okf


# ── Format memory for LLM prompt ───────────────────────────────────────────────

def format_memory_for_prompt(
    memory: dict | None,
    face_memory=None,
) -> str:
    """
    Format memory into a string for the LLM system prompt.

    Args:
        memory: The memory dict (from load_memory()).
        face_memory: Optional FaceMemory instance. If provided, known
                     people are listed with their first/last seen dates
                     and embedding counts.
    """
    if not memory:
        return ""

    lines = []

    identity  = memory.get("identity", {})
    id_fields = ["name", "age", "birthday", "city", "job", "language", "school", "nationality"]
    for field in id_fields:
        entry = identity.get(field)
        if entry:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"{field.title()}: {val}")
    for key, entry in identity.items():
        if key in id_fields:
            continue
        val = entry.get("value") if isinstance(entry, dict) else entry
        if val:
            lines.append(f"{key.replace('_', ' ').title()}: {val}")

    prefs = memory.get("preferences", {})
    if prefs:
        lines.append("")
        lines.append("Preferences:")
        for key, entry in list(prefs.items())[:15]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    projects = memory.get("projects", {})
    if projects:
        lines.append("")
        lines.append("Active Projects / Goals:")
        for key, entry in list(projects.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    rels = memory.get("relationships", {})
    if rels:
        lines.append("")
        lines.append("People in their life:")
        for key, entry in list(rels.items())[:10]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    # ── Face memory: known people ─────────────────────────────────────────
    if face_memory is not None and face_memory.is_available:
        try:
            people = face_memory.list_people()
            if people:
                lines.append("")
                lines.append("People I can recognize (face memory):")
                for p in people[:10]:
                    lines.append(
                        f"  - {p.name} (first seen: {p.first_seen}, "
                        f"embeddings: {p.embedding_count})"
                    )
        except Exception:
            pass  # Don't break the prompt if face memory fails

    wishes = memory.get("wishes", {})
    if wishes:
        lines.append("")
        lines.append("Wishes / Plans / Wants:")
        for key, entry in list(wishes.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    notes = memory.get("notes", {})
    if notes:
        lines.append("")
        lines.append("Other notes:")
        for key, entry in list(notes.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key}: {val}")

    if not lines:
        return ""

    header = "[WHAT YOU KNOW ABOUT THIS PERSON — use naturally, never recite like a list]\n"
    result = header + "\n".join(lines)
    if len(result) > 2000:
        result = result[:1997] + "…"

    return result + "\n"


# ── Backward compatibility aliases ─────────────────────────────────────────────

forget_memory = forget

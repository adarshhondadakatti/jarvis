"""
Media memory service — stores and queries images, videos, and keyframes.

This module manages the persistence of visual media and the structured
metadata associated with each frame: detected objects, detected people,
OCR text, scene descriptions, and timestamps.

Media files are stored on disk under ``memory/storage/media/`` and
tracked in the SQLite database.  Videos are decomposed into keyframes,
each of which can have its own detections and metadata.

Public API
----------
    mm = MediaMemory()
    media_id = mm.store_image(image_bytes, source="camera")
    kf_id    = mm.add_keyframe(media_id, frame_bytes, media_time=12.5)
    mm.add_person_detection(media_id, kf_id, person_id=1, confidence=0.95)
    mm.add_ocr_text(media_id, kf_id, "Hello World")

    # Queries
    results = mm.find_media_by_person(person_id=1)
    results = mm.find_media_by_object("laptop")
    results = mm.find_media_by_text("laptop")
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from memory.db import _connect, _get_lock, MEDIA_DIR


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class MediaRecord:
    """A stored media file."""
    id: int
    media_type: str
    file_path: str
    duration: Optional[float]
    width: Optional[int]
    height: Optional[int]
    created_at: str
    source: str
    sha256: str
    metadata: dict


@dataclass
class KeyframeRecord:
    """An extracted keyframe from a video."""
    id: int
    media_id: int
    media_time: float
    file_path: str
    created_at: str


# ── MediaMemory service ─────────────────────────────────────────────────────────

class MediaMemory:
    """
    Manages storage and retrieval of visual media and their metadata.

    Files are stored on disk with SHA256-based deduplication.  Each
    media file gets a database record in the ``media`` table.  Videos
    can have multiple keyframes in the ``keyframes`` table, and each
    keyframe or media file can have associated detections.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        """
        Initialise the media memory service.

        Args:
            base_dir: Root directory for media files (default: MEDIA_DIR).
        """
        self._base_dir = Path(base_dir) if base_dir else MEDIA_DIR
        self._images_dir = self._base_dir / "images"
        self._videos_dir = self._base_dir / "videos"
        self._keyframes_dir = self._base_dir / "keyframes"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create the media directory structure if it doesn't exist."""
        for d in (self._images_dir, self._videos_dir, self._keyframes_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── File management ──────────────────────────────────────────────────────

    @staticmethod
    def _sha256(data: bytes) -> str:
        """Compute SHA256 hash of bytes."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _date_dir(base: Path) -> Path:
        """Get a date-based subdirectory (e.g., 2026-07-24)."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        d = base / date_str
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_image_file(self, image_bytes: bytes, sha: str) -> str:
        """Save an image file and return its relative path."""
        date_dir = self._date_dir(self._images_dir)
        filename = f"{sha[:16]}.jpg"
        path = date_dir / filename

        # Deduplication: if file already exists, don't overwrite
        if not path.exists():
            path.write_bytes(image_bytes)

        return str(path.relative_to(self._base_dir))

    def _save_video_file(self, src_path: Path, sha: str) -> str:
        """Save a video file and return its relative path."""
        date_dir = self._date_dir(self._videos_dir)
        filename = f"{sha[:16]}.{src_path.suffix.lstrip('.')}"
        dest = date_dir / filename

        if not dest.exists():
            shutil.copy2(src_path, dest)

        return str(dest.relative_to(self._base_dir))

    def _save_keyframe(self, media_id: int, image_bytes: bytes, media_time: float) -> str:
        """Save a keyframe file and return its relative path."""
        kf_dir = self._keyframes_dir / str(media_id)
        kf_dir.mkdir(parents=True, exist_ok=True)
        # Use media_time as filename (rounded to 3 decimal places)
        filename = f"{media_time:.3f}.jpg"
        path = kf_dir / filename

        if not path.exists():
            path.write_bytes(image_bytes)

        return str(path.relative_to(self._base_dir))

    # ── Media storage ─────────────────────────────────────────────────────────

    def store_image(
        self,
        image_bytes: bytes,
        source: str = "camera",
        width: Optional[int] = None,
        height: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Store an image and create a media record.

        Args:
            image_bytes: Raw image data (JPEG/PNG).
            source: Where the image came from ("camera", "screen", "uploaded").
            width: Image width in pixels (optional).
            height: Image height in pixels (optional).
            metadata: Additional metadata as a dict (optional).

        Returns:
            The media record ID.
        """
        sha = self._sha256(image_bytes)
        file_path = self._save_image_file(image_bytes, sha)
        now = datetime.now().isoformat()

        with _get_lock():
            conn = _connect()
            try:
                # Check for existing media with same SHA256 (deduplication)
                existing = conn.execute(
                    "SELECT id FROM media WHERE sha256 = ? AND media_type = 'image'",
                    (sha,),
                ).fetchone()

                if existing:
                    return existing[0]

                meta_json = json.dumps(metadata) if metadata else None
                cur = conn.execute(
                    """
                    INSERT INTO media
                        (media_type, file_path, duration, width, height,
                         created_at, source, sha256, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("image", file_path, None, width, height, now, source, sha, meta_json),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def store_video(
        self,
        file_path: str,
        source: str = "camera",
        duration: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Store a video file and create a media record.

        The video file is copied into the media storage directory.
        Keyframes are not extracted here — call ``add_keyframe()``
        separately for each frame you want to store.

        Args:
            file_path: Path to the video file on disk.
            source: Where the video came from.
            duration: Video duration in seconds (optional).
            width: Video width in pixels (optional).
            height: Video height in pixels (optional).
            metadata: Additional metadata (codec, fps, etc.) (optional).

        Returns:
            The media record ID.
        """
        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        data = src.read_bytes()
        sha = self._sha256(data)
        rel_path = self._save_video_file(src, sha)
        now = datetime.now().isoformat()

        with _get_lock():
            conn = _connect()
            try:
                existing = conn.execute(
                    "SELECT id FROM media WHERE sha256 = ? AND media_type = 'video'",
                    (sha,),
                ).fetchone()

                if existing:
                    return existing[0]

                meta_json = json.dumps(metadata) if metadata else None
                cur = conn.execute(
                    """
                    INSERT INTO media
                        (media_type, file_path, duration, width, height,
                         created_at, source, sha256, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("video", rel_path, duration, width, height, now, source, sha, meta_json),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def add_keyframe(
        self,
        media_id: int,
        image_bytes: bytes,
        media_time: float,
    ) -> int:
        """
        Add a keyframe to a video media record.

        Args:
            media_id: The video's media ID.
            image_bytes: The keyframe image data (JPEG/PNG).
            media_time: Timestamp within the video (seconds).

        Returns:
            The keyframe record ID.
        """
        file_path = self._save_keyframe(media_id, image_bytes, media_time)
        now = datetime.now().isoformat()

        with _get_lock():
            conn = _connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO keyframes
                        (media_id, media_time, file_path, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (media_id, media_time, file_path, now),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    # ── Detection storage ────────────────────────────────────────────────────

    def add_object_detection(
        self,
        media_id: int,
        label: str,
        confidence: float,
        bbox: Optional[tuple[int, int, int, int]] = None,
        keyframe_id: Optional[int] = None,
    ) -> int:
        """
        Record an object detection on a media file or keyframe.

        Args:
            media_id: The media record ID.
            label: Object label (e.g., "laptop", "chair").
            confidence: Detection confidence (0.0 – 1.0).
            bbox: Bounding box as (x1, y1, x2, y2) (optional).
            keyframe_id: If the detection is on a keyframe (optional).

        Returns:
            The detection record ID.
        """
        bbox_json = json.dumps(bbox) if bbox else None
        now = datetime.now().isoformat()

        with _get_lock():
            conn = _connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO detected_objects
                        (media_id, keyframe_id, label, confidence, bbox, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (media_id, keyframe_id, label, confidence, bbox_json, now),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def add_person_detection(
        self,
        media_id: int,
        person_id: int,
        confidence: float,
        bbox: Optional[tuple[int, int, int, int]] = None,
        keyframe_id: Optional[int] = None,
    ) -> int:
        """
        Record a person detection on a media file or keyframe.

        Args:
            media_id: The media record ID.
            person_id: The recognized person's ID (from people table).
            confidence: Recognition confidence (0.0 – 1.0).
            bbox: Bounding box as (x1, y1, x2, y2) (optional).
            keyframe_id: If the detection is on a keyframe (optional).

        Returns:
            The detection record ID.
        """
        bbox_json = json.dumps(bbox) if bbox else None
        now = datetime.now().isoformat()

        with _get_lock():
            conn = _connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO detected_people
                        (media_id, keyframe_id, person_id, confidence, bbox, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (media_id, keyframe_id, person_id, confidence, bbox_json, now),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def add_ocr_text(
        self,
        media_id: int,
        text: str,
        bbox: Optional[tuple[int, int, int, int]] = None,
        keyframe_id: Optional[int] = None,
    ) -> int:
        """
        Record OCR text extracted from a media file or keyframe.

        Args:
            media_id: The media record ID.
            text: The extracted text.
            bbox: Bounding box of the text (optional).
            keyframe_id: If the text is from a keyframe (optional).

        Returns:
            The OCR record ID.
        """
        bbox_json = json.dumps(bbox) if bbox else None
        now = datetime.now().isoformat()

        with _get_lock():
            conn = _connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO ocr_text
                        (media_id, keyframe_id, text, bbox, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (media_id, keyframe_id, text, bbox_json, now),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def add_scene_description(
        self,
        media_id: int,
        description: str,
        model: Optional[str] = None,
        keyframe_id: Optional[int] = None,
    ) -> int:
        """
        Record a scene description for a media file or keyframe.

        Args:
            media_id: The media record ID.
            description: The scene description text.
            model: Which model generated the description (optional).
            keyframe_id: If the description is for a keyframe (optional).

        Returns:
            The scene description record ID.
        """
        now = datetime.now().isoformat()

        with _get_lock():
            conn = _connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO scene_descriptions
                        (media_id, keyframe_id, description, model, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (media_id, keyframe_id, description, model, now),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_media(self, media_id: int) -> Optional[MediaRecord]:
        """Get a media record by ID."""
        with _get_lock():
            conn = _connect()
            try:
                row = conn.execute(
                    "SELECT * FROM media WHERE id = ?", (media_id,)
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return None

        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        return MediaRecord(
            id=row["id"],
            media_type=row["media_type"],
            file_path=row["file_path"],
            duration=row["duration"],
            width=row["width"],
            height=row["height"],
            created_at=row["created_at"],
            source=row["source"],
            sha256=row["sha256"],
            metadata=metadata,
        )

    def get_keyframes(self, media_id: int) -> list[KeyframeRecord]:
        """Get all keyframes for a media record."""
        with _get_lock():
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, media_id, media_time, file_path, created_at
                    FROM keyframes
                    WHERE media_id = ?
                    ORDER BY media_time
                    """,
                    (media_id,),
                ).fetchall()
            finally:
                conn.close()

        return [
            KeyframeRecord(
                id=row["id"],
                media_id=row["media_id"],
                media_time=row["media_time"],
                file_path=row["file_path"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def find_media_by_person(self, person_id: int) -> list[dict]:
        """
        Find all media containing a specific person.

        Returns a list of dicts with media info and detection details,
        sorted by creation date (most recent first).
        """
        with _get_lock():
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT
                        m.id, m.media_type, m.file_path, m.created_at,
                        m.duration, m.width, m.height,
                        dp.confidence, dp.bbox,
                        k.media_time, k.id as keyframe_id
                    FROM detected_people dp
                    JOIN media m ON m.id = dp.media_id
                    LEFT JOIN keyframes k ON k.id = dp.keyframe_id
                    WHERE dp.person_id = ?
                    ORDER BY m.created_at DESC
                    """,
                    (person_id,),
                ).fetchall()
            finally:
                conn.close()

        return self._format_media_results(rows)

    def find_media_by_object(self, label: str) -> list[dict]:
        """
        Find all media containing a specific object label.

        Args:
            label: Object label to search for (case-insensitive).

        Returns:
            List of dicts with media info and detection details.
        """
        with _get_lock():
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT
                        m.id, m.media_type, m.file_path, m.created_at,
                        m.duration, m.width, m.height,
                        do.confidence, do.bbox,
                        k.media_time, k.id as keyframe_id
                    FROM detected_objects do
                    JOIN media m ON m.id = do.media_id
                    LEFT JOIN keyframes k ON k.id = do.keyframe_id
                    WHERE do.label = ?
                    ORDER BY m.created_at DESC
                    """,
                    (label,),
                ).fetchall()
            finally:
                conn.close()

        return self._format_media_results(rows)

    def find_media_by_text(self, query: str) -> list[dict]:
        """
        Find all media containing OCR text matching the query.

        Uses SQLite's LIKE operator for substring matching.

        Args:
            query: Text to search for in OCR-extracted text.

        Returns:
            List of dicts with media info and OCR text.
        """
        with _get_lock():
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT
                        m.id, m.media_type, m.file_path, m.created_at,
                        m.duration, m.width, m.height,
                        o.text, o.bbox,
                        k.media_time, k.id as keyframe_id
                    FROM ocr_text o
                    JOIN media m ON m.id = o.media_id
                    LEFT JOIN keyframes k ON k.id = o.keyframe_id
                    WHERE o.text LIKE ?
                    ORDER BY m.created_at DESC
                    """,
                    (f"%{query}%",),
                ).fetchall()
            finally:
                conn.close()

        results = []
        for row in rows:
            bbox = None
            if row["bbox"]:
                try:
                    bbox = json.loads(row["bbox"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append({
                "media_id": row["id"],
                "media_type": row["media_type"],
                "file_path": row["file_path"],
                "created_at": row["created_at"],
                "duration": row["duration"],
                "width": row["width"],
                "height": row["height"],
                "ocr_text": row["text"],
                "bbox": bbox,
                "media_time": row["media_time"],
                "keyframe_id": row["keyframe_id"],
            })
        return results

    def find_media_by_person_and_person(
        self,
        person_id_a: int,
        person_id_b: int,
    ) -> list[dict]:
        """
        Find all media containing both specified people.

        This enables queries like "Find videos containing both Alice and Charlie."

        Args:
            person_id_a: First person's ID.
            person_id_b: Second person's ID.

        Returns:
            List of dicts with media info.
        """
        with _get_lock():
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT DISTINCT
                        m.id, m.media_type, m.file_path, m.created_at,
                        m.duration, m.width, m.height
                    FROM media m
                    WHERE m.id IN (
                        SELECT media_id FROM detected_people WHERE person_id = ?
                    )
                    AND m.id IN (
                        SELECT media_id FROM detected_people WHERE person_id = ?
                    )
                    ORDER BY m.created_at DESC
                    """,
                    (person_id_a, person_id_b),
                ).fetchall()
            finally:
                conn.close()

        return [
            {
                "media_id": row["id"],
                "media_type": row["media_type"],
                "file_path": row["file_path"],
                "created_at": row["created_at"],
                "duration": row["duration"],
                "width": row["width"],
                "height": row["height"],
            }
            for row in rows
        ]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _format_media_results(self, rows: list) -> list[dict]:
        """Format database rows into result dicts."""
        results = []
        for row in rows:
            bbox = None
            if row["bbox"]:
                try:
                    bbox = json.loads(row["bbox"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append({
                "media_id": row["id"],
                "media_type": row["media_type"],
                "file_path": row["file_path"],
                "created_at": row["created_at"],
                "duration": row["duration"],
                "width": row["width"],
                "height": row["height"],
                "confidence": row["confidence"],
                "bbox": bbox,
                "media_time": row["media_time"],
                "keyframe_id": row["keyframe_id"],
            })
        return results

    def get_full_path(self, rel_path: str) -> Path:
        """Convert a relative media path to an absolute filesystem path."""
        return self._base_dir / rel_path

    def delete_media(self, media_id: int) -> bool:
        """
        Delete a media record and its associated files.

        Args:
            media_id: The media record ID.

        Returns:
            True if the media was deleted, False if not found.
        """
        with _get_lock():
            conn = _connect()
            try:
                # Get the file path before deleting the record
                row = conn.execute(
                    "SELECT file_path FROM media WHERE id = ?", (media_id,)
                ).fetchone()

                if row is None:
                    return False

                # Delete the file from disk
                file_path = self._base_dir / row["file_path"]
                if file_path.exists():
                    file_path.unlink()

                # Delete the media record (FK cascade handles keyframes,
                # detections, OCR, scene descriptions)
                conn.execute("DELETE FROM media WHERE id = ?", (media_id,))
                conn.commit()
                return True
            finally:
                conn.close()

    def count_media(self) -> int:
        """Return the total number of stored media files."""
        with _get_lock():
            conn = _connect()
            try:
                return conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]
            finally:
                conn.close()

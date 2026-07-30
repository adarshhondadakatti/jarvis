"""
Face memory service — manages person enrollment and recognition.

This module ties together three layers:
    1. FaceBackend (recognition/face_backend.py) — detection + embedding
    2. SQLiteVectorStore (memory/vector_store.py) — embedding storage
    3. db.py — person records, sightings, and metadata

Public API
----------
    fm = FaceMemory()
    person_id = fm.enroll_person("Alice", image_bytes)
    matches   = fm.recognize_faces(image_bytes, min_confidence=0.6)
    history   = fm.get_person_history(person_id)
    people    = fm.list_people()

The service is designed to be used by the main JARVIS loop: when a user
says "This is Alice", the camera captures a frame and enroll_person() is
called.  Later, whenever a camera frame contains a face, recognize_faces()
is called to identify who it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from memory.db import _connect, _lock, MEDIA_DIR
from memory.vector_store import VectorStore, get_vector_store
from recognition.face_backend import FaceBackend, FaceDetection, InsightFaceError


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Person:
    """A known person in the face memory system."""
    id: int
    name: str
    first_seen: str
    last_seen: str
    embedding_count: int = 0
    profile: str = "default"
    metadata: dict = field(default_factory=dict)


@dataclass
class RecognitionResult:
    """
    Result of recognizing a single face in an image.

    If the face matches a known person, ``person_id`` and ``person_name``
    are set.  If the face is unknown, they are None.
    ``profile`` is the memory profile associated with the recognized person.
    """
    face: FaceDetection
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    profile: str = "default"
    confidence: float = 0.0
    is_known: bool = False


# ── FaceMemory service ─────────────────────────────────────────────────────────

class FaceMemory:
    """
    Manages person enrollment, recognition, and history.

    Uses InsightFace for face detection and embedding extraction,
    SQLiteVectorStore for embedding similarity search, and the
    SQLite database for person records and sighting history.
    """

    #: Default cosine similarity threshold for recognizing a known person.
    #: InsightFace embeddings: same-person > 0.6, different-person < 0.4.
    DEFAULT_MIN_CONFIDENCE: float = 0.6

    #: When recognition confidence exceeds this threshold, a new embedding
    #: is automatically saved to improve future recognition.
    AUTO_ENROLL_CONFIDENCE: float = 0.85

    def __init__(
        self,
        backend: Optional[FaceBackend] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        """
        Initialise the face memory service.

        Args:
            backend: Face recognition backend (default: InsightFace FaceBackend).
            vector_store: Vector store for embeddings (default: singleton).
        """
        self._backend = backend
        self._vector_store = vector_store or get_vector_store()
        self._backend_available = False

    # ── Backend management ───────────────────────────────────────────────────

    def _ensure_backend(self) -> FaceBackend:
        """Lazily create the face backend on first use."""
        if self._backend is not None:
            return self._backend
        try:
            self._backend = FaceBackend()
            self._backend_available = True
        except InsightFaceError:
            self._backend_available = False
            raise
        return self._backend

    @property
    def is_available(self) -> bool:
        """Return True if the face recognition backend is available."""
        try:
            self._ensure_backend()
            return True
        except InsightFaceError:
            return False

    # ── Person management ────────────────────────────────────────────────────

    def enroll_person(
        self,
        name: str,
        image_bytes: bytes,
        source: str = "camera",
        profile: str = "default",
    ) -> Optional[int]:
        """
        Enroll a new person by name using a face from the given image.

        This is the "This is Alice" flow:
            1. Detect faces in the image
            2. Extract embeddings for each face
            3. Create a person record in the database
            4. Store the embedding(s) in the vector store
            5. Record a sighting event

        If multiple faces are detected, the largest face is used.
        If no faces are detected, returns None.

        Args:
            name: The person's name.
            image_bytes: Raw image data (JPEG/PNG) containing a face.
            source: Where the image came from ("camera", "photo", etc.).
            profile: Memory profile to associate with this person (for
                     automatic profile switching on recognition).

        Returns:
            The new person's ID, or None if no face was found.
        """
        backend = self._ensure_backend()
        faces = backend.detect_and_embed(image_bytes)

        if not faces:
            return None

        # Use the largest face (first in the list, since detect_and_embed
        # returns faces sorted by size)
        face = faces[0]
        if face.embedding is None:
            return None

        now = datetime.now().isoformat()

        with _lock:
            conn = _connect()
            try:
                # Check if a person with this name already exists
                existing = conn.execute(
                    "SELECT id FROM people WHERE name = ?", (name,)
                ).fetchone()

                if existing:
                    person_id = existing[0]
                    # Update last_seen and profile
                    conn.execute(
                        "UPDATE people SET last_seen = ?, profile = ? WHERE id = ?",
                        (now, profile, person_id),
                    )
                else:
                    # Create new person
                    cur = conn.execute(
                        """
                        INSERT INTO people
                            (name, first_seen, last_seen, embedding_dim, profile, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (name, now, now, backend.embedding_dim, profile, None),
                    )
                    person_id = cur.lastrowid

                conn.commit()
            finally:
                conn.close()

        # Store the embedding in the vector store
        self._vector_store.add(
            person_id=person_id,
            embedding=face.embedding,
            source=source,
            confidence=face.confidence,
        )

        # Record a sighting in detected_people (if we have media context)
        # This is called from the media layer when a full image is stored

        return person_id

    def update_person(
        self,
        person_id: int,
        image_bytes: bytes,
        source: str = "camera",
    ) -> int:
        """
        Add additional embeddings to an existing person.

        Used to improve recognition accuracy over time by collecting
        multiple embeddings from different angles, lighting, etc.

        Args:
            person_id: The person's ID.
            image_bytes: Image containing the person's face.
            source: Where the image came from.

        Returns:
            Number of new embeddings added.
        """
        backend = self._ensure_backend()
        faces = backend.detect_and_embed(image_bytes)

        added = 0
        now = datetime.now().isoformat()

        with _lock:
            conn = _connect()
            try:
                for face in faces:
                    if face.embedding is None:
                        continue
                    self._vector_store.add(
                        person_id=person_id,
                        embedding=face.embedding,
                        source=source,
                        confidence=face.confidence,
                    )
                    added += 1

                if added > 0:
                    conn.execute(
                        "UPDATE people SET last_seen = ? WHERE id = ?",
                        (now, person_id),
                    )
                    conn.commit()
            finally:
                conn.close()

        return added

    def delete_person(self, person_id: int) -> bool:
        """
        Delete a person and all their embeddings.

        Args:
            person_id: The person's ID.

        Returns:
            True if the person was deleted, False if not found.
        """
        with _lock:
            conn = _connect()
            try:
                # Delete embeddings first (FK cascade would handle this,
                # but explicit is clearer)
                self._vector_store.delete_for_person(person_id)

                cur = conn.execute(
                    "DELETE FROM people WHERE id = ?", (person_id,)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def list_people(self) -> list[Person]:
        """Return all known people, sorted by last_seen descending."""
        with _lock:
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT p.id, p.name, p.first_seen, p.last_seen,
                           COUNT(pe.id) as embedding_count
                    FROM people p
                    LEFT JOIN person_embeddings pe ON pe.person_id = p.id
                    GROUP BY p.id
                    ORDER BY p.last_seen DESC
                    """
                ).fetchall()
            finally:
                conn.close()

        return [
            Person(
                id=row["id"],
                name=row["name"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                embedding_count=row["embedding_count"],
            )
            for row in rows
        ]

    def get_person(self, person_id: int) -> Optional[Person]:
        """Get a single person by ID."""
        with _lock:
            conn = _connect()
            try:
                row = conn.execute(
                    """
                    SELECT p.id, p.name, p.first_seen, p.last_seen,
                           COUNT(pe.id) as embedding_count,
                           p.metadata
                    FROM people p
                    LEFT JOIN person_embeddings pe ON pe.person_id = p.id
                    WHERE p.id = ?
                    GROUP BY p.id
                    """,
                    (person_id,),
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return None

        import json
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        return Person(
            id=row["id"],
            name=row["name"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            embedding_count=row["embedding_count"],
            metadata=metadata,
        )

    def find_person_by_name(self, name: str) -> Optional[Person]:
        """Find a person by name (case-insensitive)."""
        with _lock:
            conn = _connect()
            try:
                row = conn.execute(
                    """
                    SELECT p.id, p.name, p.first_seen, p.last_seen, p.profile,
                           COUNT(pe.id) as embedding_count
                    FROM people p
                    LEFT JOIN person_embeddings pe ON pe.person_id = p.id
                    WHERE p.name = ?
                    GROUP BY p.id
                    """,
                    (name,),
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return None

        return Person(
            id=row["id"],
            name=row["name"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            embedding_count=row["embedding_count"],
            profile=row["profile"] or "default",
        )

    # ── Recognition ────────────────────────────────────────────────────────────

    def recognize_faces(
        self,
        image_bytes: bytes,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        auto_enroll: bool = True,
    ) -> list[RecognitionResult]:
        """
        Detect and recognize all faces in an image.

        For each detected face:
            1. Compute the embedding
            2. Search the vector store for the best match
            3. If similarity >= min_confidence, the face is identified
            4. If similarity >= AUTO_ENROLL_CONFIDENCE and auto_enroll is True,
               the embedding is saved to improve future recognition

        Args:
            image_bytes: Raw image data (JPEG/PNG).
            min_confidence: Minimum cosine similarity to identify a face.
            auto_enroll: Whether to save high-confidence embeddings.

        Returns:
            List of RecognitionResult objects, one per detected face.
        """
        backend = self._ensure_backend()
        faces = backend.detect_and_embed(image_bytes)

        results = []
        for face in faces:
            result = RecognitionResult(face=face)

            if face.embedding is not None:
                matches = self._vector_store.search(
                    face.embedding,
                    top_k=1,
                    min_similarity=min_confidence,
                )

                if matches:
                    best = matches[0]
                    person = self.get_person(best.person_id)
                    if person:
                        result.person_id = person.id
                        result.person_name = person.name
                        result.profile = person.profile
                        result.confidence = best.similarity
                        result.is_known = True

                        # Auto-enroll: save high-confidence embeddings
                        if auto_enroll and best.similarity >= self.AUTO_ENROLL_CONFIDENCE:
                            self._vector_store.add(
                                person_id=person.id,
                                embedding=face.embedding,
                                source="camera",
                                confidence=face.confidence,
                            )
                            # Update last_seen
                            now = datetime.now().isoformat()
                            with _lock:
                                conn = _connect()
                                try:
                                    conn.execute(
                                        "UPDATE people SET last_seen = ? WHERE id = ?",
                                        (now, person.id),
                                    )
                                    conn.commit()
                                finally:
                                    conn.close()

            results.append(result)

        return results

    # ── History ────────────────────────────────────────────────────────────────

    def get_person_history(
        self,
        person_id: int,
        limit: int = 50,
    ) -> list[dict]:
        """
        Get all sightings of a person across media.

        Returns a list of dicts with:
            - media_type, file_path, created_at, confidence, bbox
            - keyframe info (if applicable)
        """
        with _lock:
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT
                        m.media_type,
                        m.file_path,
                        m.created_at as media_created,
                        dp.confidence,
                        dp.bbox,
                        k.media_time,
                        k.created_at as keyframe_created
                    FROM detected_people dp
                    JOIN media m ON m.id = dp.media_id
                    LEFT JOIN keyframes k ON k.id = dp.keyframe_id
                    WHERE dp.person_id = ?
                    ORDER BY dp.created_at DESC
                    LIMIT ?
                    """,
                    (person_id, limit),
                ).fetchall()
            finally:
                conn.close()

        import json
        history = []
        for row in rows:
            bbox = None
            if row["bbox"]:
                try:
                    bbox = json.loads(row["bbox"])
                except (json.JSONDecodeError, TypeError):
                    pass
            history.append({
                "media_type": row["media_type"],
                "file_path": row["file_path"],
                "media_created": row["media_created"],
                "confidence": row["confidence"],
                "bbox": bbox,
                "media_time": row["media_time"],
                "keyframe_created": row["keyframe_created"],
            })
        return history

    def get_last_seen(
        self,
        person_id: int,
    ) -> Optional[dict]:
        """
        Get the most recent sighting of a person.

        Returns a dict with media info, or None if the person has
        never been seen in any stored media.
        """
        history = self.get_person_history(person_id, limit=1)
        return history[0] if history else None

    # ── Utility ────────────────────────────────────────────────────────────────

    def count_people(self) -> int:
        """Return the total number of known people."""
        with _lock:
            conn = _connect()
            try:
                return conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
            finally:
                conn.close()

    def count_embeddings(self) -> int:
        """Return the total number of stored embeddings."""
        return self._vector_store.count()

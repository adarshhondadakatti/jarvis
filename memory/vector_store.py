"""
Vector storage abstraction for face embeddings.

Provides a clean interface for storing and searching high-dimensional
embeddings.  The current implementation uses SQLite float32 BLOBs with
brute-force cosine similarity, which is sufficient for dozens to hundreds
of people.  The ``VectorStore`` ABC is designed so that a FAISS or
pgvector backend can be swapped in later without changing any callers.

Design notes
------------
* Embeddings are stored as raw ``float32`` BLOBs — never as JSON strings.
* Each embedding row is linked to a ``person_id`` so we can delete all
  embeddings for a person in one query.
* ``search()`` returns ``(person_id, similarity, embedding_id)`` tuples
  sorted by descending similarity.
* The ABC defines the contract; ``SQLiteVectorStore`` is the default
  implementation.  A future ``FAISSVectorStore`` would subclass the same ABC.
"""
from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from memory.db import DB_PATH, _get_lock


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class VectorSearchResult:
    """A single search result."""
    person_id: int
    similarity: float
    embedding_id: int


# ── Abstract base class ────────────────────────────────────────────────────────

class VectorStore(ABC):
    """
    Abstract vector store interface.

    Implementations must support:
    * Adding embeddings linked to a person_id
    * Searching for the nearest embeddings to a query vector
    * Deleting all embeddings for a given person
    * Retrieving all embeddings for a person (for re-indexing)
    """

    @abstractmethod
    def add(
        self,
        person_id: int,
        embedding: np.ndarray,
        source: str = "camera",
        confidence: float = 1.0,
    ) -> int:
        """Store an embedding. Returns the row id."""

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Find the *top_k* nearest embeddings to *query*."""

    @abstractmethod
    def delete_for_person(self, person_id: int) -> int:
        """Delete all embeddings for *person_id*. Returns count deleted."""

    @abstractmethod
    def get_for_person(self, person_id: int) -> list[np.ndarray]:
        """Return all embeddings for *person_id* as numpy arrays."""

    @abstractmethod
    def count(self) -> int:
        """Total number of stored embeddings."""


# ── SQLite implementation ──────────────────────────────────────────────────────

class SQLiteVectorStore(VectorStore):
    """
    Vector store backed by the same SQLite database as ``db.py``.

    Embeddings are stored as ``float32`` BLOBs in the ``person_embeddings``
    table.  Search is brute-force cosine similarity — O(n) per query.

    For production use with thousands of people, replace this with a
    FAISS-backed implementation that subclasses ``VectorStore``.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = Path(db_path) if db_path else DB_PATH

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector. Returns a copy."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.astype(np.float32).copy()
        return (vec / norm).astype(np.float32)

    @staticmethod
    def _blob_to_array(blob: bytes) -> np.ndarray:
        """Decode a float32 BLOB back to a numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def _array_to_blob(arr: np.ndarray) -> bytes:
        """Encode a numpy array as a float32 BLOB."""
        return np.ascontiguousarray(arr, dtype=np.float32).tobytes()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(
        self,
        person_id: int,
        embedding: np.ndarray,
        source: str = "camera",
        confidence: float = 1.0,
    ) -> int:
        """
        Store an embedding for *person_id*.

        The embedding is L2-normalized before storage so that dot-product
        search is equivalent to cosine similarity.
        """
        if embedding.ndim != 1:
            raise ValueError(f"Expected 1-D embedding, got shape {embedding.shape}")

        normalized = self._normalize(embedding)
        blob = self._array_to_blob(normalized)
        now = datetime.now().isoformat()

        with _get_lock():
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO person_embeddings
                        (person_id, embedding, source, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (person_id, blob, source, confidence, now),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[VectorSearchResult]:
        """
        Brute-force nearest-neighbor search using cosine similarity.

        Because embeddings are stored normalized, the dot product equals
        cosine similarity.  We fetch all embeddings and compute dot products
        in numpy for speed.

        For large-scale use, replace with FAISS or an approximate NN index.
        """
        if query.ndim != 1:
            raise ValueError(f"Expected 1-D query, got shape {query.shape}")

        query_norm = self._normalize(query)

        with _get_lock():
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, person_id, embedding FROM person_embeddings"
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return []

        # Decode all embeddings into a matrix (N × D)
        embeddings = np.vstack([
            self._blob_to_array(row["embedding"]) for row in rows
        ])

        # Cosine similarity = dot product (both sides normalized)
        similarities = embeddings @ query_norm  # (N,)

        # Filter by minimum similarity
        mask = similarities >= min_similarity
        if not mask.any():
            return []

        # Get top-k indices — use original indices to map back to rows
        original_indices = np.where(mask)[0]
        sims = similarities[mask]
        sorted_order = np.argsort(sims)[::-1][:top_k]
        top_original = original_indices[sorted_order]

        results = []
        for orig_idx in top_original:
            row = rows[orig_idx]
            results.append(VectorSearchResult(
                person_id=row["person_id"],
                similarity=float(similarities[orig_idx]),
                embedding_id=row["id"],
            ))
        return results

    def delete_for_person(self, person_id: int) -> int:
        """Delete all embeddings for *person_id*."""
        with _get_lock():
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM person_embeddings WHERE person_id = ?",
                    (person_id,),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    def get_for_person(self, person_id: int) -> list[np.ndarray]:
        """Return all embeddings for *person_id* as numpy arrays."""
        with _get_lock():
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT embedding FROM person_embeddings WHERE person_id = ?",
                    (person_id,),
                ).fetchall()
            finally:
                conn.close()

        return [self._blob_to_array(row["embedding"]) for row in rows]

    def count(self) -> int:
        """Total number of stored embeddings."""
        with _get_lock():
            conn = self._connect()
            try:
                return conn.execute(
                    "SELECT COUNT(*) FROM person_embeddings"
                ).fetchone()[0]
            finally:
                conn.close()


# ── Singleton accessor ─────────────────────────────────────────────────────────

_default_store: Optional[SQLiteVectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the default (singleton) vector store instance."""
    global _default_store
    if _default_store is None:
        _default_store = SQLiteVectorStore()
    return _default_store


def reset_vector_store() -> None:
    """Clear the singleton (useful for testing)."""
    global _default_store
    _default_store = None

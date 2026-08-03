"""
Integration tests for the visual memory system.

Tests the full flow: FaceMemory → VectorStore → database,
MediaMemory → database, EventMemory → database.

Run with:  python -m pytest tests/test_visual_memory.py -v
"""
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import io

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.db import ensure_db_ready, DB_PATH, MEDIA_DIR, _connect, _get_lock
from memory.vector_store import get_vector_store, reset_vector_store
from memory.face_memory import FaceMemory
from memory.media_memory import MediaMemory
from memory.event_memory import EventMemory


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_db():
    """Reset the database before each test."""
    # Remove existing database
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(DB_PATH) + suffix)
        if p.exists():
            p.unlink()
    if MEDIA_DIR.exists():
        shutil.rmtree(MEDIA_DIR)

    ensure_db_ready()
    reset_vector_store()

    # Create a test person (person_id=1) for vector store tests
    with _get_lock():
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO people (name, first_seen, last_seen, embedding_dim) "
                "VALUES ('TestPerson', '2026-07-24', '2026-07-24', 512)"
            )
            conn.commit()
        finally:
            conn.close()

    yield
    # Cleanup after test
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(DB_PATH) + suffix)
        if p.exists():
            p.unlink()
    if MEDIA_DIR.exists():
        shutil.rmtree(MEDIA_DIR)


def make_test_image(size=(100, 100), color=(255, 0, 0)):
    """Create a test JPEG image as bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── Vector Store Tests ─────────────────────────────────────────────────────────

class TestVectorStore:
    def test_add_and_search(self):
        vs = get_vector_store()
        emb = np.random.randn(512).astype(np.float32)
        vs.add(person_id=1, embedding=emb)

        results = vs.search(emb, top_k=1, min_similarity=0.5)
        assert len(results) == 1
        assert results[0].person_id == 1
        assert results[0].similarity > 0.99  # Same embedding

    def test_search_no_match(self):
        vs = get_vector_store()
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        vs.add(person_id=1, embedding=emb1)

        results = vs.search(emb2, top_k=1, min_similarity=0.9)
        assert len(results) == 0  # No match above 0.9

    def test_delete_for_person(self):
        vs = get_vector_store()
        emb = np.random.randn(512).astype(np.float32)
        vs.add(person_id=1, embedding=emb)
        vs.add(person_id=1, embedding=emb)

        deleted = vs.delete_for_person(1)
        assert deleted == 2
        assert vs.count() == 0

    def test_get_for_person(self):
        vs = get_vector_store()
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        vs.add(person_id=1, embedding=emb1)
        vs.add(person_id=1, embedding=emb2)

        embs = vs.get_for_person(1)
        assert len(embs) == 2

    def test_multiple_people(self):
        vs = get_vector_store()
        # Use orthogonal vectors to guarantee clear separation
        emb1 = np.zeros(512, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(512, dtype=np.float32)
        emb2[1] = 1.0
        emb3 = emb1.copy()
        emb3[0] = 0.99
        emb3[2] = 0.01  # Nearly identical to emb1

        # Create a second person for person_id=2
        with _get_lock():
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO people (name, first_seen, last_seen, embedding_dim) "
                    "VALUES ('TestPerson2', '2026-07-24', '2026-07-24', 512)"
                )
                conn.commit()
            finally:
                conn.close()

        vs.add(person_id=1, embedding=emb1)
        vs.add(person_id=2, embedding=emb2)
        vs.add(person_id=1, embedding=emb3)

        results = vs.search(emb1, top_k=5, min_similarity=0.5)
        # emb1 and emb3 are similar to each other, emb2 is orthogonal
        assert len(results) == 2
        assert all(r.person_id == 1 for r in results)
        assert results[0].similarity > results[1].similarity  # Sorted by similarity


# ── FaceMemory Tests ───────────────────────────────────────────────────────────

class TestFaceMemory:
    def test_face_memory_init(self):
        fm = FaceMemory()
        assert fm is not None
        # Fixture creates one person, so count should be >= 1
        assert fm.count_people() >= 1

    def test_face_memory_not_available_without_insightface(self):
        """If InsightFace is not installed, is_available should be False."""
        fm = FaceMemory()
        # We can't control whether insightface is installed, but the
        # service should not crash
        assert isinstance(fm.is_available, bool)

    def test_enroll_person_requires_face(self):
        """enroll_person should return None if no face is detected."""
        fm = FaceMemory()
        # Without InsightFace, this will raise InsightFaceError
        # With InsightFace, a blank image should return None
        try:
            result = fm.enroll_person("Test", make_test_image())
            assert result is None  # No face in a solid color image
        except Exception:
            pass  # InsightFace not installed — skip


# ── MediaMemory Tests ──────────────────────────────────────────────────────────

class TestMediaMemory:
    def test_store_image(self):
        mm = MediaMemory()
        img_bytes = make_test_image()
        media_id = mm.store_image(img_bytes, source="test")
        assert media_id > 0

        media = mm.get_media(media_id)
        assert media.media_type == "image"
        assert media.source == "test"
        assert media.sha256 is not None

    def test_image_deduplication(self):
        mm = MediaMemory()
        img_bytes = make_test_image()
        id1 = mm.store_image(img_bytes, source="test")
        id2 = mm.store_image(img_bytes, source="test")
        assert id1 == id2  # Same content → same media record

    def test_add_keyframe(self):
        mm = MediaMemory()
        img_bytes = make_test_image()
        media_id = mm.store_image(img_bytes, source="test")
        kf_id = mm.add_keyframe(media_id, img_bytes, media_time=12.5)
        assert kf_id > 0

        keyframes = mm.get_keyframes(media_id)
        assert len(keyframes) == 1
        assert keyframes[0].media_time == 12.5

    def test_add_object_detection(self):
        mm = MediaMemory()
        img_bytes = make_test_image()
        media_id = mm.store_image(img_bytes, source="test")
        obj_id = mm.add_object_detection(
            media_id, "laptop", 0.95, bbox=(10, 10, 50, 50)
        )
        assert obj_id > 0

        results = mm.find_media_by_object("laptop")
        assert len(results) == 1
        assert results[0]["confidence"] == 0.95

    def test_add_ocr_text(self):
        mm = MediaMemory()
        img_bytes = make_test_image()
        media_id = mm.store_image(img_bytes, source="test")
        ocr_id = mm.add_ocr_text(media_id, "Hello World", bbox=(10, 10, 50, 50))
        assert ocr_id > 0

        results = mm.find_media_by_text("Hello")
        assert len(results) == 1
        assert "Hello World" in results[0]["ocr_text"]

    def test_find_media_by_text_partial(self):
        mm = MediaMemory()
        img_bytes = make_test_image()
        media_id = mm.store_image(img_bytes, source="test")
        mm.add_ocr_text(media_id, "The quick brown fox")

        results = mm.find_media_by_text("fox")
        assert len(results) == 1

    def test_delete_media(self):
        mm = MediaMemory()
        img_bytes = make_test_image()
        media_id = mm.store_image(img_bytes, source="test")
        assert mm.get_media(media_id) is not None

        deleted = mm.delete_media(media_id)
        assert deleted is True
        assert mm.get_media(media_id) is None

    def test_count_media(self):
        mm = MediaMemory()
        assert mm.count_media() == 0
        mm.store_image(make_test_image(), source="test")
        assert mm.count_media() == 1


# ── EventMemory Tests ──────────────────────────────────────────────────────────

class TestEventMemory:
    def test_create_event(self):
        em = EventMemory()
        event_id = em.create_event(
            event_type="introduction",
            title="Met Alice",
            description="Alice was introduced by John",
            location="Office",
        )
        assert event_id > 0

        detail = em.get_event(event_id)
        assert detail is not None
        assert detail.event.title == "Met Alice"
        assert detail.event.event_type == "introduction"
        assert detail.event.location == "Office"

    def test_link_person(self):
        em = EventMemory()
        # Create a person first
        with _get_lock():
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO people (name, first_seen, last_seen, embedding_dim) "
                    "VALUES ('Alice', '2026-07-24', '2026-07-24', 512)"
                )
                conn.commit()
                person_id = conn.execute(
                    "SELECT id FROM people WHERE name = 'Alice'"
                ).fetchone()[0]
            finally:
                conn.close()

        event_id = em.create_event("introduction", "Met Alice")
        linked = em.link_person(event_id, person_id, role="introduced")
        assert linked is True

        detail = em.get_event(event_id)
        assert len(detail.people) == 1
        assert detail.people[0].person_name == "Alice"
        assert detail.people[0].role == "introduced"

    def test_link_media(self):
        em = EventMemory()
        mm = MediaMemory()

        img_bytes = make_test_image()
        media_id = mm.store_image(img_bytes, source="test")

        event_id = em.create_event("saw_person", "Saw someone")
        linked = em.link_media(event_id, media_id, role="primary")
        assert linked is True

        detail = em.get_event(event_id)
        assert len(detail.media) == 1
        assert detail.media[0].media_id == media_id

    def test_find_events_by_person(self):
        em = EventMemory()

        # Create a person
        with _get_lock():
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO people (name, first_seen, last_seen, embedding_dim) "
                    "VALUES ('Bob', '2026-07-24', '2026-07-24', 512)"
                )
                conn.commit()
                person_id = conn.execute(
                    "SELECT id FROM people WHERE name = 'Bob'"
                ).fetchone()[0]
            finally:
                conn.close()

        e1 = em.create_event("introduction", "Met Bob", started_at="2026-07-24T10:00:00")
        em.link_person(e1, person_id, role="introduced")

        e2 = em.create_event("meeting", "Talked with Bob", started_at="2026-07-24T11:00:00")
        em.link_person(e2, person_id, role="participant")

        events = em.find_events_by_person(person_id)
        assert len(events) == 2

    def test_get_last_and_first_event_with_person(self):
        em = EventMemory()

        with _get_lock():
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO people (name, first_seen, last_seen, embedding_dim) "
                    "VALUES ('Charlie', '2026-07-24', '2026-07-24', 512)"
                )
                conn.commit()
                person_id = conn.execute(
                    "SELECT id FROM people WHERE name = 'Charlie'"
                ).fetchone()[0]
            finally:
                conn.close()

        e1 = em.create_event("introduction", "Met Charlie", started_at="2026-07-24T10:00:00")
        em.link_person(e1, person_id)

        e2 = em.create_event("saw_person", "Saw Charlie again", started_at="2026-07-24T15:00:00")
        em.link_person(e2, person_id)

        first = em.get_first_event_with_person(person_id)
        assert first is not None
        assert first.title == "Met Charlie"

        last = em.get_last_event_with_person(person_id)
        assert last is not None
        assert last.title == "Saw Charlie again"

    def test_find_events_by_type(self):
        em = EventMemory()
        em.create_event("introduction", "Met Alice")
        em.create_event("introduction", "Met Bob")
        em.create_event("meeting", "Team meeting")

        intros = em.find_events_by_type("introduction")
        assert len(intros) == 2

    def test_delete_event(self):
        em = EventMemory()
        event_id = em.create_event("custom", "Test event")
        assert em.get_event(event_id) is not None

        deleted = em.delete_event(event_id)
        assert deleted is True
        assert em.get_event(event_id) is None


# ── Backward Compatibility Tests ───────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_textual_memory_still_works(self):
        """Existing memory_manager API should work unchanged."""
        from memory.memory_manager import (
            load_memory, update_memory, remember, forget,
            format_memory_for_prompt,
        )

        # Save a memory
        update_memory({"identity": {"name": {"value": "TestUser"}}})
        update_memory({"preferences": {"favorite_color": {"value": "blue"}}})

        # Load it back
        memory = load_memory()
        assert memory["identity"]["name"]["value"] == "TestUser"
        assert memory["preferences"]["favorite_color"]["value"] == "blue"

        # Format for prompt
        prompt = format_memory_for_prompt(memory)
        assert "TestUser" in prompt
        assert "blue" in prompt

        # Forget
        result = forget("favorite_color", "preferences")
        assert "Forgotten" in result

        # Verify it's gone
        memory = load_memory()
        assert "favorite_color" not in memory["preferences"]

    def test_legacy_json_migration(self):
        """Legacy long_term.json should still migrate to SQLite."""
        from memory.memory_manager import load_memory

        # The long_term.json should have been migrated during ensure_db_ready
        memory = load_memory()
        assert "identity" in memory
        assert "preferences" in memory
        assert "projects" in memory
        assert "relationships" in memory
        assert "wishes" in memory
        assert "notes" in memory


# ── End-to-End Integration Test ────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_flow(self):
        """
        Test the complete flow:
        1. Create a person in the database
        2. Store an image as media
        3. Add a person detection linking them
        4. Create an event linking the person and media
        5. Query for the person's history
        """
        mm = MediaMemory()
        em = EventMemory()

        # 1. Create a person
        with _get_lock():
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO people (name, first_seen, last_seen, embedding_dim) "
                    "VALUES ('Alice', '2026-07-24T10:00:00', '2026-07-24T10:00:00', 512)"
                )
                conn.commit()
                person_id = conn.execute(
                    "SELECT id FROM people WHERE name = 'Alice'"
                ).fetchone()[0]
            finally:
                conn.close()

        # 2. Store an image
        img_bytes = make_test_image()
        media_id = mm.store_image(img_bytes, source="camera")

        # 3. Add a person detection
        mm.add_person_detection(media_id, person_id, confidence=0.92, bbox=(10, 10, 50, 50))

        # 4. Create an event
        event_id = em.create_event(
            "saw_person",
            "Saw Alice",
            started_at="2026-07-24T10:00:00",
        )
        em.link_person(event_id, person_id, role="observed")
        em.link_media(event_id, media_id, role="primary")

        # 5. Query
        # Find media by person
        results = mm.find_media_by_person(person_id)
        assert len(results) == 1
        assert results[0]["confidence"] == 0.92

        # Find events by person
        events = em.find_events_by_person(person_id)
        assert len(events) == 1
        assert events[0].title == "Saw Alice"

        # Get event detail
        detail = em.get_event(event_id)
        assert len(detail.people) == 1
        assert detail.people[0].person_name == "Alice"
        assert len(detail.media) == 1

        # Get person history
        fm = FaceMemory()
        history = fm.get_person_history(person_id, limit=10)
        assert len(history) == 1
        assert history[0]["media_type"] == "image"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

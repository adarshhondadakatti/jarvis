"""
Event memory service — links people, media, and text into episodic events.

This module provides the "episodic memory" layer of JARVIS's memory system.
Events represent meaningful occurrences (introductions, meetings, sightings)
and link together:
    - People (who was involved)
    - Media (images, videos, keyframes that document the event)
    - Text memories (contextual facts)
    - Timestamps and locations

This enables queries like:
    "When did you last see Alice?"     → find_events_by_person(alice_id)
    "When did you first meet Bob?"     → find_events_by_type("introduction")
    "Show me when John entered the room" → find_events_by_person(john_id)
    "Find videos containing both Alice and Charlie" → media layer

Public API
----------
    em = EventMemory()
    event_id = em.create_event("introduction", "Met Alice", "Alice was introduced by John")
    em.link_person(event_id, person_id=1, role="introduced")
    em.link_media(event_id, media_id=1, role="primary")
    events = em.find_events_by_person(person_id=1)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from memory.db import _connect, _get_lock


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class EventRecord:
    """An episodic memory event."""
    id: int
    event_type: str
    title: str
    description: Optional[str]
    started_at: str
    ended_at: Optional[str]
    location: Optional[str]
    metadata: dict
    created_at: str


@dataclass
class EventPersonLink:
    """A person linked to an event."""
    person_id: int
    person_name: str
    role: Optional[str]


@dataclass
class EventMediaLink:
    """A media file linked to an event."""
    media_id: int
    media_type: str
    file_path: str
    role: Optional[str]


@dataclass
class EventDetail:
    """An event with all its linked people and media."""
    event: EventRecord
    people: list[EventPersonLink] = field(default_factory=list)
    media: list[EventMediaLink] = field(default_factory=list)


# ── EventMemory service ─────────────────────────────────────────────────────────

class EventMemory:
    """
    Manages episodic memory events and their relationships.

    Events are the central linking mechanism between people, media, and
    text memories.  Each event has a type, title, description, timestamp,
    and optional location.  People and media are linked via junction tables
    with optional roles (e.g., "introduced", "participant", "primary").
    """

    # ── Event type constants ───────────────────────────────────────────────

    TYPE_INTRODUCTION = "introduction"
    TYPE_MEETING = "meeting"
    TYPE_SAW_PERSON = "saw_person"
    TYPE_CONVERSATION = "conversation"
    TYPE_CUSTOM = "custom"

    VALID_TYPES = frozenset({
        TYPE_INTRODUCTION,
        TYPE_MEETING,
        TYPE_SAW_PERSON,
        TYPE_CONVERSATION,
        TYPE_CUSTOM,
    })

    def __init__(self) -> None:
        """Initialise the event memory service."""
        pass

    # ── Event creation ─────────────────────────────────────────────────────

    def create_event(
        self,
        event_type: str,
        title: str,
        description: Optional[str] = None,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        location: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Create a new episodic memory event.

        Args:
            event_type: Type of event (use TYPE_* constants or "custom").
            title: Short, human-readable title.
            description: Longer description (optional).
            started_at: ISO timestamp when the event started
                        (default: now).
            ended_at: ISO timestamp when the event ended (optional).
            location: Where the event occurred (optional).
            metadata: Additional structured data (optional).

        Returns:
            The new event's ID.
        """
        if event_type not in self.VALID_TYPES:
            event_type = self.TYPE_CUSTOM

        if started_at is None:
            started_at = datetime.now().isoformat()

        now = datetime.now().isoformat()
        meta_json = json.dumps(metadata) if metadata else None

        with _get_lock():
            conn = _connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO events
                        (event_type, title, description, started_at,
                         ended_at, location, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (event_type, title, description, started_at,
                     ended_at, location, meta_json, now),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def update_event(
        self,
        event_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        ended_at: Optional[str] = None,
        location: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Update an existing event's fields.

        Only non-None fields are updated.
        """
        updates = []
        values = []

        if title is not None:
            updates.append("title = ?")
            values.append(title)
        if description is not None:
            updates.append("description = ?")
            values.append(description)
        if ended_at is not None:
            updates.append("ended_at = ?")
            values.append(ended_at)
        if location is not None:
            updates.append("location = ?")
            values.append(location)
        if metadata is not None:
            updates.append("metadata = ?")
            values.append(json.dumps(metadata))

        if not updates:
            return False

        values.append(event_id)

        with _get_lock():
            conn = _connect()
            try:
                cur = conn.execute(
                    f"UPDATE events SET {', '.join(updates)} WHERE id = ?",
                    values,
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def delete_event(self, event_id: int) -> bool:
        """
        Delete an event and all its links.

        Args:
            event_id: The event's ID.

        Returns:
            True if the event was deleted, False if not found.
        """
        with _get_lock():
            conn = _connect()
            try:
                cur = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ── Linking ────────────────────────────────────────────────────────────

    def link_media(
        self,
        event_id: int,
        media_id: int,
        role: Optional[str] = None,
    ) -> bool:
        """
        Link a media file to an event.

        Args:
            event_id: The event's ID.
            media_id: The media file's ID.
            role: Role of the media in the event (e.g., "primary",
                  "context", "result").

        Returns:
            True if the link was created, False if event or media not found.
        """
        with _get_lock():
            conn = _connect()
            try:
                # Verify event exists
                event_exists = conn.execute(
                    "SELECT 1 FROM events WHERE id = ?", (event_id,)
                ).fetchone()
                if not event_exists:
                    return False

                # Verify media exists
                media_exists = conn.execute(
                    "SELECT 1 FROM media WHERE id = ?", (media_id,)
                ).fetchone()
                if not media_exists:
                    return False

                conn.execute(
                    """
                    INSERT OR REPLACE INTO event_media (event_id, media_id, role)
                    VALUES (?, ?, ?)
                    """,
                    (event_id, media_id, role),
                )
                conn.commit()
                return True
            finally:
                conn.close()

    def link_person(
        self,
        event_id: int,
        person_id: int,
        role: Optional[str] = None,
    ) -> bool:
        """
        Link a person to an event.

        Args:
            event_id: The event's ID.
            person_id: The person's ID.
            role: Role of the person in the event (e.g., "participant",
                  "introduced", "observer").

        Returns:
            True if the link was created, False if event or person not found.
        """
        with _get_lock():
            conn = _connect()
            try:
                # Verify event exists
                event_exists = conn.execute(
                    "SELECT 1 FROM events WHERE id = ?", (event_id,)
                ).fetchone()
                if not event_exists:
                    return False

                # Verify person exists
                person_exists = conn.execute(
                    "SELECT 1 FROM people WHERE id = ?", (person_id,)
                ).fetchone()
                if not person_exists:
                    return False

                conn.execute(
                    """
                    INSERT OR REPLACE INTO event_people (event_id, person_id, role)
                    VALUES (?, ?, ?)
                    """,
                    (event_id, person_id, role),
                )
                conn.commit()
                return True
            finally:
                conn.close()

    def unlink_media(self, event_id: int, media_id: int) -> bool:
        """Remove a media link from an event."""
        with _get_lock():
            conn = _connect()
            try:
                cur = conn.execute(
                    "DELETE FROM event_media WHERE event_id = ? AND media_id = ?",
                    (event_id, media_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def unlink_person(self, event_id: int, person_id: int) -> bool:
        """Remove a person link from an event."""
        with _get_lock():
            conn = _connect()
            try:
                cur = conn.execute(
                    "DELETE FROM event_people WHERE event_id = ? AND person_id = ?",
                    (event_id, person_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ── Queries ────────────────────────────────────────────────────────────

    def get_event(self, event_id: int) -> Optional[EventDetail]:
        """
        Get an event with all its linked people and media.

        Args:
            event_id: The event's ID.

        Returns:
            EventDetail with all links, or None if not found.
        """
        with _get_lock():
            conn = _connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, event_type, title, description, started_at,
                           ended_at, location, metadata, created_at
                    FROM events
                    WHERE id = ?
                    """,
                    (event_id,),
                ).fetchone()

                if row is None:
                    return None

                # Get linked people
                people_rows = conn.execute(
                    """
                    SELECT ep.person_id, p.name, ep.role
                    FROM event_people ep
                    JOIN people p ON p.id = ep.person_id
                    WHERE ep.event_id = ?
                    """,
                    (event_id,),
                ).fetchall()

                # Get linked media
                media_rows = conn.execute(
                    """
                    SELECT em.media_id, m.media_type, m.file_path, em.role
                    FROM event_media em
                    JOIN media m ON m.id = em.media_id
                    WHERE em.event_id = ?
                    """,
                    (event_id,),
                ).fetchall()
            finally:
                conn.close()

        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        event = EventRecord(
            id=row["id"],
            event_type=row["event_type"],
            title=row["title"],
            description=row["description"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            location=row["location"],
            metadata=metadata,
            created_at=row["created_at"],
        )

        people = [
            EventPersonLink(
                person_id=r["person_id"],
                person_name=r["name"],
                role=r["role"],
            )
            for r in people_rows
        ]

        media = [
            EventMediaLink(
                media_id=r["media_id"],
                media_type=r["media_type"],
                file_path=r["file_path"],
                role=r["role"],
            )
            for r in media_rows
        ]

        return EventDetail(event=event, people=people, media=media)

    def find_events_by_type(
        self,
        event_type: str,
        limit: int = 50,
    ) -> list[EventRecord]:
        """
        Find events of a specific type.

        Args:
            event_type: The event type (use TYPE_* constants).
            limit: Maximum number of results.

        Returns:
            List of EventRecord, sorted by started_at descending.
        """
        with _get_lock():
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, event_type, title, description, started_at,
                           ended_at, location, metadata, created_at
                    FROM events
                    WHERE event_type = ?
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (event_type, limit),
                ).fetchall()
            finally:
                conn.close()

        return [self._row_to_event(row) for row in rows]

    def find_events_by_person(
        self,
        person_id: int,
        limit: int = 50,
    ) -> list[EventRecord]:
        """
        Find all events involving a specific person.

        Args:
            person_id: The person's ID.
            limit: Maximum number of results.

        Returns:
            List of EventRecord, sorted by started_at descending.
        """
        with _get_lock():
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT e.id, e.event_type, e.title, e.description,
                           e.started_at, e.ended_at, e.location,
                           e.metadata, e.created_at
                    FROM events e
                    JOIN event_people ep ON ep.event_id = e.id
                    WHERE ep.person_id = ?
                    ORDER BY e.started_at DESC
                    LIMIT ?
                    """,
                    (person_id, limit),
                ).fetchall()
            finally:
                conn.close()

        return [self._row_to_event(row) for row in rows]

    def find_events_by_time(
        self,
        start: str,
        end: Optional[str] = None,
        limit: int = 50,
    ) -> list[EventRecord]:
        """
        Find events within a time range.

        Args:
            start: ISO timestamp for the start of the range.
            end: ISO timestamp for the end of the range (default: now).
            limit: Maximum number of results.

        Returns:
            List of EventRecord, sorted by started_at descending.
        """
        if end is None:
            end = datetime.now().isoformat()

        with _get_lock():
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, event_type, title, description, started_at,
                           ended_at, location, metadata, created_at
                    FROM events
                    WHERE started_at >= ? AND started_at <= ?
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (start, end, limit),
                ).fetchall()
            finally:
                conn.close()

        return [self._row_to_event(row) for row in rows]

    def find_events_by_person_and_person(
        self,
        person_id_a: int,
        person_id_b: int,
        limit: int = 50,
    ) -> list[EventRecord]:
        """
        Find events where both specified people were present.

        Args:
            person_id_a: First person's ID.
            person_id_b: Second person's ID.
            limit: Maximum number of results.

        Returns:
            List of EventRecord, sorted by started_at descending.
        """
        with _get_lock():
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT DISTINCT
                        e.id, e.event_type, e.title, e.description,
                        e.started_at, e.ended_at, e.location,
                        e.metadata, e.created_at
                    FROM events e
                    WHERE e.id IN (
                        SELECT event_id FROM event_people WHERE person_id = ?
                    )
                    AND e.id IN (
                        SELECT event_id FROM event_people WHERE person_id = ?
                    )
                    ORDER BY e.started_at DESC
                    LIMIT ?
                    """,
                    (person_id_a, person_id_b, limit),
                ).fetchall()
            finally:
                conn.close()

        return [self._row_to_event(row) for row in rows]

    def get_last_event_with_person(
        self,
        person_id: int,
    ) -> Optional[EventRecord]:
        """
        Get the most recent event involving a person.

        Useful for answering "When did you last see Alice?"
        """
        events = self.find_events_by_person(person_id, limit=1)
        return events[0] if events else None

    def get_first_event_with_person(
        self,
        person_id: int,
    ) -> Optional[EventRecord]:
        """
        Get the first event involving a person.

        Useful for answering "When did you first meet Bob?"
        """
        with _get_lock():
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT e.id, e.event_type, e.title, e.description,
                           e.started_at, e.ended_at, e.location,
                           e.metadata, e.created_at
                    FROM events e
                    JOIN event_people ep ON ep.event_id = e.id
                    WHERE ep.person_id = ?
                    ORDER BY e.started_at ASC
                    LIMIT 1
                    """,
                    (person_id,),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return None
        return self._row_to_event(rows[0])

    def count_events(self) -> int:
        """Return the total number of events."""
        with _get_lock():
            conn = _connect()
            try:
                return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            finally:
                conn.close()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_event(row) -> EventRecord:
        """Convert a database row to an EventRecord."""
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        return EventRecord(
            id=row["id"],
            event_type=row["event_type"],
            title=row["title"],
            description=row["description"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            location=row["location"],
            metadata=metadata,
            created_at=row["created_at"],
        )

"""
JARVIS memory package — persistent storage and retrieval of memories.

Textual memory (existing):
    load_memory, save_memory, update_memory, remember, forget,
    format_memory_for_prompt, init_memory

Visual memory (new):
    FaceMemory  — person enrollment and face recognition
    MediaMemory — image/video/keyframe storage and queries
    EventMemory — episodic memory linking people, media, and events
    VectorStore — abstract embedding storage (SQLite now, FAISS later)

Face recognition backend:
    FaceBackend — InsightFace face detection + embedding extraction
"""
# ── Textual memory (existing API, unchanged) ───────────────────────────────────
from memory.memory_manager import (
    init_memory,
    load_memory,
    save_memory,
    update_memory,
    remember,
    forget,
    forget_memory,
    format_memory_for_prompt,
)

# ── Visual memory (new) ────────────────────────────────────────────────────────
from memory.face_memory import FaceMemory, Person, RecognitionResult
from memory.media_memory import MediaMemory, MediaRecord, KeyframeRecord
from memory.event_memory import EventMemory, EventRecord, EventDetail
from memory.vector_store import VectorStore, SQLiteVectorStore, get_vector_store

# ── Face recognition backend ───────────────────────────────────────────────────
from recognition.face_backend import FaceBackend, FaceDetection, InsightFaceError
from recognition.face_quality import FaceQuality

__all__ = [
    # Textual memory
    "init_memory",
    "load_memory",
    "save_memory",
    "update_memory",
    "remember",
    "forget",
    "forget_memory",
    "format_memory_for_prompt",
    # Visual memory
    "FaceMemory",
    "Person",
    "RecognitionResult",
    "MediaMemory",
    "MediaRecord",
    "KeyframeRecord",
    "EventMemory",
    "EventRecord",
    "EventDetail",
    "VectorStore",
    "SQLiteVectorStore",
    "get_vector_store",
    # Face recognition backend
    "FaceBackend",
    "FaceDetection",
    "InsightFaceError",
    "FaceQuality",
]

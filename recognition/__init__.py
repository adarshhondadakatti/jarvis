"""
Recognition package — modular computer vision backends.

Each backend provides a clean interface for a specific recognition task
(face detection, object detection, OCR, etc.) and can be swapped or
extended independently of the memory layer.

Currently available:
    FaceBackend      — InsightFace face detection + embedding extraction
    FaceQuality     — AdaFace-inspired quality assessment and adaptive thresholds
"""
from recognition.face_backend import FaceBackend, FaceDetection, InsightFaceError
from recognition.face_quality import FaceQuality

__all__ = ["FaceBackend", "FaceDetection", "InsightFaceError", "FaceQuality"]

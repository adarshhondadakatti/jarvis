"""
Face recognition backend using InsightFace.

Provides face detection and 512-dimensional embedding extraction using
the InsightFace library with the buffalo_l model.  Designed to work on
CPU (no GPU required) and to be swappable with other backends.

Interface
---------
    backend = FaceBackend()
    faces = backend.detect_and_embed(image_bytes)
    # faces: list[FaceDetection] with .bbox, .confidence, .embedding

The backend is lazily initialised — the InsightFace model is only loaded
on first use, not at import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── Optional dependency check ──────────────────────────────────────────────────

try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False


class InsightFaceError(ImportError):
    """Raised when InsightFace is not installed or the model fails to load."""


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    """
    A single detected face.

    Attributes:
        bbox:      Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        confidence: Detection confidence (0.0 – 1.0).
        embedding: 512-dim face embedding (float32), or None if not computed.
        face_image: Cropped face image as numpy array (H, W, 3) RGB, or None.
    """
    bbox: tuple[int, int, int, int]
    confidence: float
    embedding: Optional[np.ndarray] = None
    face_image: Optional[np.ndarray] = None

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


# ── Backend ────────────────────────────────────────────────────────────────────

class FaceBackend:
    """
    Face detection and embedding extraction using InsightFace.

    Uses the buffalo_l model with:
    * RetinaFace for face detection (with 5-point landmark alignment)
    * ArcFace R100 for 512-dim embedding extraction

    The model is loaded lazily on first use to avoid blocking import time.
    """

    #: Dimensionality of the embeddings produced by the recognition model.
    EMBEDDING_DIM: int = 512

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: Optional[list[str]] = None,
    ) -> None:
        """
        Initialise the face backend.

        Args:
            model_name: InsightFace model pack name.
            providers: ONNX Runtime providers (default: CPU only).
        """
        if not _INSIGHTFACE_AVAILABLE:
            raise InsightFaceError(
                "insightface is not installed.  Run:  pip install insightface"
            )

        self._model_name = model_name
        self._providers = providers or ["CPUExecutionProvider"]
        self._app: Optional[FaceAnalysis] = None

    # ── Lazy initialisation ───────────────────────────────────────────────────

    def _ensure_loaded(self) -> FaceAnalysis:
        """Load the InsightFace model on first use."""
        if self._app is not None:
            return self._app

        app = FaceAnalysis(
            name=self._model_name,
            providers=self._providers,
        )
        app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
        self._app = app
        return app

    # ── Image decoding ────────────────────────────────────────────────────────

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        """
        Decode raw image bytes (JPEG/PNG) into an RGB numpy array.

        Uses cv2 if available, falls back to PIL.
        """
        try:
            import cv2
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            # cv2 returns BGR; convert to RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return np.array(img)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_faces(self, image_bytes: bytes) -> list[FaceDetection]:
        """
        Detect faces in an image without computing embeddings.

        Args:
            image_bytes: Raw image data (JPEG/PNG).

        Returns:
            List of FaceDetection objects (embedding will be None).
        """
        app = self._ensure_loaded()
        img = self._decode_image(image_bytes)
        faces = app.get(img)  # largest faces first (sorted below)

        # Sort by face area (largest first)
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            results.append(FaceDetection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(face.det_score),
                embedding=None,
                face_image=None,
            ))
        return results

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract a face embedding from a cropped face image.

        Args:
            face_image: Cropped face as numpy array (H, W, 3) RGB.

        Returns:
            512-dim float32 embedding.
        """
        app = self._ensure_loaded()
        # InsightFace's get() handles alignment internally, but for a
        # pre-cropped face we use the recognition model directly.
        # We use the app's model to get the embedding.
        faces = app.get(face_image)
        if not faces:
            raise ValueError("No face detected in the cropped image")
        return faces[0].embedding.astype(np.float32)

    def detect_and_embed(self, image_bytes: bytes) -> list[FaceDetection]:
        """
        Detect faces and compute embeddings in a single pass.

        This is the primary method for face recognition — it runs the
        full InsightFace pipeline (detection → alignment → embedding).

        Args:
            image_bytes: Raw image data (JPEG/PNG).

        Returns:
            List of FaceDetection objects with embeddings populated,
            sorted by bounding box size (largest first).
        """
        app = self._ensure_loaded()
        img = self._decode_image(image_bytes)
        faces = app.get(img)

        # Sort by face area (largest first)
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Crop the face region
            face_img = img[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None

            results.append(FaceDetection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(face.det_score),
                embedding=face.embedding.astype(np.float32) if face.embedding is not None else None,
                face_image=face_img,
            ))
        return results

    def is_available(self) -> bool:
        """Return True if InsightFace is installed and the model loaded."""
        return _INSIGHTFACE_AVAILABLE

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embeddings produced by this backend."""
        return self.EMBEDDING_DIM

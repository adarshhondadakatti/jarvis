"""
Face image quality assessment and adaptive thresholding.

Implements the AdaFace insight that the L2 norm of a face embedding
(before L2-normalisation) is a strong proxy for image quality -- no
separate quality estimator network is required.

Paper: "AdaFace: Quality Adaptive Margin for Face Recognition"
       Kim, Jain, Liu — CVPR 2022
       https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.pdf

At inference time this means we can make recognition thresholds adaptive:
  * low-quality query faces  → raise the similarity bar (adaptive_threshold)
  * norms below a floor       → reject the embedding entirely during enrollment

  Note: the similarity score itself is **not** scaled by quality at
  inference time.  AdaFace uses quality to adjust the loss-margin
  during training; at recognition time we keep the raw cosine
  similarity and only raise the decision threshold for degraded
  inputs.  The ``adjusted_similarity`` / ``quality_factor`` methods
  below exist for experimental purposes but must NOT be used as the
  match decision criterion — doing so multiplies every similarity by
  ~0.5 (typical quality score), causing all matches to fail.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class FaceQuality:
    """
    Quality assessment and adaptive thresholding for face recognition.

    The **quality score** is derived from the raw embedding norm:

        quality = clip((||z|| - q_min) / (q_max - q_min), 0, 1)

    Embeddings from well-lit, frontal, in-frame faces typically have norms
    in the 14–33 range with the InsightFace ArcFace R100 model; norms below
    ``q_min`` indicate severely degraded inputs (blur, occlusion, extreme
    pose, poor lighting).
    """

    #: Default base threshold for auto-enrollment (overridable per-call).
    AUTO_ENROLL_THRESHOLD: float = 0.85
    #: Maximum quality penalty applied to the auto-enroll threshold.
    MAX_ENROLL_PENALTY: float = 0.10

    def __init__(
        self,
        q_min: float = 10.0,
        q_max: float = 35.0,
        base_threshold: float = 0.6,
        max_penalty: float = 0.15,
        min_norm_floor: float = 5.0,
    ) -> None:
        """
        Args:
            q_min:            Embedding norm below which quality is 0.
            q_max:            Embedding norm above which quality is 1.
            base_threshold:   Base cosine similarity for recognition
                              (quality = 1.0 reproduces the old fixed threshold).
            max_penalty:      Maximum upward adjustment to the threshold for
                              the lowest-quality faces.
            min_norm_floor:   Hard floor below which an embedding is considered
                              unusable for enrollment.
        """
        self.q_min = q_min
        self.q_max = q_max
        self.base_threshold = base_threshold
        self.max_penalty = max_penalty
        self.min_norm_floor = min_norm_floor

    # ── Quality scoring ──────────────────────────────────────────────────────────

    def quality_score(self, embedding: np.ndarray) -> float:
        """
        Map an embedding's L2 norm to a 0–1 quality score.

        Args:
            embedding: Raw (pre-normalisation) face embedding.

        Returns:
            Quality score in [0, 1].  1.0 = high quality, 0.0 = unusable.
        """
        norm = float(np.linalg.norm(embedding))
        return self.quality_from_norm(norm)

    def quality_from_norm(self, norm: float) -> float:
        """Map a raw embedding norm directly to a quality score."""
        span = self.q_max - self.q_min
        if span <= 0:
            return 1.0
        return float(np.clip((norm - self.q_min) / span, 0.0, 1.0))

    # ── Adaptive thresholds ──────────────────────────────────────────────────────

    def adaptive_threshold(self, quality: float, base_threshold: Optional[float] = None) -> float:
        """
        Quality-adaptive recognition threshold.

        Low-quality faces require a higher cosine similarity to be
        considered a match, preventing false positives from degraded inputs.

        Args:
            quality:         Quality score in [0, 1].
            base_threshold:  Override for the base threshold (defaults to
                             ``self.base_threshold``).  Allows callers to
                             use a custom ``min_confidence`` as the base.

        Returns:
            Effective cosine-similarity threshold.
        """
        quality = float(np.clip(quality, 0.0, 1.0))
        base = base_threshold if base_threshold is not None else self.base_threshold
        return base + self.max_penalty * (1.0 - quality)

    def adaptive_enroll_threshold(
        self,
        quality: float,
        base_threshold: Optional[float] = None,
    ) -> float:
        """
        Quality-adaptive threshold for auto-enrollment.

        Only high-confidence, high-quality embeddings are automatically
        saved.  Low-quality faces must achieve an even higher similarity
        to trigger auto-enrollment.

        Args:
            quality:         Quality score in [0, 1].
            base_threshold:  Override for the base auto-enroll threshold
                             (defaults to ``AUTO_ENROLL_THRESHOLD``).

        Returns:
            Effective cosine-similarity threshold for auto-enrollment.
        """
        quality = float(np.clip(quality, 0.0, 1.0))
        base = base_threshold if base_threshold is not None else self.AUTO_ENROLL_THRESHOLD
        return base + self.MAX_ENROLL_PENALTY * (1.0 - quality)

    # ── Quality-weighted scoring ─────────────────────────────────────────────────

    def quality_factor(
        self,
        query_quality: float,
        reference_quality: float,
    ) -> float:
        """
        Multiplicative weight for similarity based on combined quality.

        .. warning::
            **Do not use this for face-recognition matching.**  Multiplying
            the cosine similarity by the geometric mean of quality scores
            (which for norm ≈ 22 yields ≈ 0.48) crushes legitimate
            same-person matches so that 0.80 becomes 0.38 — failing every
            threshold.  The correct approach is to compare the **raw**
            cosine similarity against an :meth:`adaptive_threshold`
            (quality raised for degraded faces).

        Uses the geometric mean of the query and reference quality scores:
        a good-quality query matched against a good-quality reference gets
        full weight; a degraded query or reference down-weights the score.

        Args:
            query_quality:    Quality of the query face embedding.
            reference_quality: Quality of the stored reference embedding.

        Returns:
            Weight in [0, 1].
        """
        q = float(np.clip(query_quality, 0.0, 1.0))
        r = float(np.clip(reference_quality, 0.0, 1.0))
        if q <= 0 or r <= 0:
            return 0.0
        return float(np.sqrt(q * r))

    def adjusted_similarity(
        self,
        raw_similarity: float,
        query_quality: float,
        reference_quality: float,
    ) -> float:
        """
        Apply the quality factor to a raw cosine similarity.

        .. warning::
            **Not used in production face recognition.**  This method
            multiplies the cosine similarity by the quality factor, which
            destroys match scores for typical embedding norms (~22 → quality
            ≈ 0.48 → factor ≈ 0.48).  Use raw cosine similarity with
            :meth:`adaptive_threshold` instead, as done in
            ``FaceMemory.recognize_faces``.

        The adjusted score reflects not just geometric proximity but
        also how trustworthy both embeddings are.
        """
        factor = self.quality_factor(query_quality, reference_quality)
        return float(raw_similarity * factor)

    # ── Embedding usability ──────────────────────────────────────────────────────

    def is_embedding_usable(self, embedding: np.ndarray) -> bool:
        """
        Check whether an embedding is good enough to store during enrollment.

        Embeddings with a norm below ``min_norm_floor`` are too noisy to
        contribute to recognition and should be discarded.
        """
        norm = float(np.linalg.norm(embedding))
        return norm >= self.min_norm_floor

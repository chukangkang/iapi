import asyncio
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image

from app.config import Settings
from app.restoration.codeformer_service import FaceCandidate, FaceCandidateResult


logger = logging.getLogger(__name__)


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if len(left_values) != len(right_values):
        raise ValueError("ArcFace embeddings must have the same dimensions")
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ValueError("ArcFace embedding has zero-length norm")
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


class ArcFaceIdentityService:
    """Score CodeFormer candidates against their aligned source faces using ArcFace."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._analyzer: Optional[Any] = None
        self._analyzer_key: Optional[tuple[str, str, tuple[str, ...], int]] = None

    async def score_candidates(self, result: FaceCandidateResult) -> FaceCandidateResult:
        if not self.settings.insightface_enabled or not result.candidates:
            return result
        try:
            return await asyncio.to_thread(self._score_sync, result)
        except Exception as exc:
            logger.warning("ArcFace identity scoring failed; rejecting unscored candidates: %s", exc)
            return FaceCandidateResult(
                source_image=result.source_image,
                candidates=tuple(replace(candidate, identity_score=None, identity_accepted=False) for candidate in result.candidates),
                detected_face_count=result.detected_face_count,
            )

    def _score_sync(self, result: FaceCandidateResult) -> FaceCandidateResult:
        analyzer = self._get_analyzer()
        scored_candidates = []
        for candidate in result.candidates:
            try:
                source_embedding, source_landmarks = self._features(analyzer, candidate.original_face)
                candidate_embedding, candidate_landmarks = self._features(analyzer, candidate.restored_face)
                score = cosine_similarity(source_embedding, candidate_embedding)
                accepted = score >= self.settings.insightface_identity_threshold
                scored_candidates.append(
                    replace(
                        candidate,
                        identity_score=score,
                        identity_accepted=accepted,
                        original_aligned_landmarks=source_landmarks,
                        restored_aligned_landmarks=candidate_landmarks,
                    )
                )
            except Exception as exc:
                logger.warning("ArcFace failed for face %s; rejecting candidate: %s", candidate.face_index, exc)
                scored_candidates.append(replace(candidate, identity_score=None, identity_accepted=False))
        return FaceCandidateResult(
            source_image=result.source_image,
            candidates=tuple(scored_candidates),
            detected_face_count=result.detected_face_count,
        )

    def _features(
        self, analyzer: Any, image: Image.Image
    ) -> tuple[Iterable[float], tuple[tuple[float, float], ...]]:
        faces = analyzer.get(self._image_to_bgr(image))
        if not faces:
            raise ValueError("InsightFace did not detect a face in the aligned crop")
        face = max(faces, key=self._face_area)
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            raise ValueError("InsightFace face result is missing an ArcFace embedding")
        landmarks = getattr(face, "kps", None)
        if landmarks is None or len(landmarks) != 5:
            raise ValueError("InsightFace face result is missing five landmarks")
        normalized_landmarks = tuple(
            (float(point[0]), float(point[1])) for point in landmarks
        )
        return embedding, normalized_landmarks

    @staticmethod
    def _face_area(face: Any) -> float:
        bbox = getattr(face, "bbox", (0, 0, 0, 0))
        return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

    @staticmethod
    def _image_to_bgr(image: Image.Image) -> Any:
        import numpy as np

        return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()

    def _get_analyzer(self) -> Any:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        available = set(ort.get_available_providers())
        providers = []
        if self.settings.device != "cpu" and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
        key = (
            self.settings.insightface_model_name,
            str(Path(self.settings.insightface_model_root).resolve()),
            tuple(providers),
            ctx_id,
        )
        if self._analyzer is not None and self._analyzer_key == key:
            return self._analyzer
        analyzer = FaceAnalysis(
            name=self.settings.insightface_model_name,
            root=self.settings.insightface_model_root,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        analyzer.prepare(ctx_id=ctx_id, det_size=(512, 512))
        self._analyzer = analyzer
        self._analyzer_key = key
        return analyzer
import logging
from dataclasses import replace

from PIL import Image, ImageFilter, ImageStat

from app.config import Settings
from app.restoration.codeformer_service import FaceCandidate, FaceCandidateResult


logger = logging.getLogger(__name__)


def detail_score(image: Image.Image) -> float:
    """Return a bounded high-frequency detail estimate for an aligned face."""
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    mean = ImageStat.Stat(edges).mean[0] / 255.0
    return max(0.0, min(1.0, mean * 2.0))


class FaceCandidateSelector:
    """Apply hard safety gates, rank each face candidate, and select safe improvements."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def select(self, result: FaceCandidateResult) -> FaceCandidateResult:
        selected_candidates = []
        for candidate in result.candidates:
            hard_rejection = self._hard_rejection_reason(candidate)
            if hard_rejection is not None:
                selected_candidates.append(
                    replace(
                        candidate,
                        quality_score=None,
                        composite_score=None,
                        selected=False,
                        rejection_reason=hard_rejection,
                    )
                )
                continue
            try:
                quality = self._quality_score(candidate)
                composite = self._composite_score(candidate, quality)
                accepted = composite >= self.settings.face_candidate_min_score
                selected_candidates.append(
                    replace(
                        candidate,
                        quality_score=quality,
                        composite_score=composite,
                        selected=accepted,
                        rejection_reason=None if accepted else "score",
                    )
                )
            except Exception as exc:
                logger.warning("Candidate scoring failed for face %s; falling back: %s", candidate.face_index, exc)
                selected_candidates.append(
                    replace(
                        candidate,
                        quality_score=None,
                        composite_score=None,
                        selected=False,
                        rejection_reason="scoring_error",
                    )
                )
        return FaceCandidateResult(
            source_image=result.source_image,
            candidates=tuple(selected_candidates),
            detected_face_count=result.detected_face_count,
        )

    @staticmethod
    def _hard_rejection_reason(candidate: FaceCandidate) -> str | None:
        if not candidate.identity_accepted or candidate.identity_score is None:
            return "identity"
        if not candidate.landmark_accepted:
            return "landmarks"
        return None

    def _quality_score(self, candidate: FaceCandidate) -> float:
        original_detail = detail_score(candidate.original_face)
        restored_detail = detail_score(candidate.restored_face)
        # Neutral is 0.5; only measurable detail gain moves the score upward.
        return max(0.0, min(1.0, 0.5 + (restored_detail - original_detail)))

    def _composite_score(self, candidate: FaceCandidate, quality: float) -> float:
        identity = max(0.0, min(1.0, float(candidate.identity_score or 0.0)))
        geometry = self._geometry_score(candidate)
        detection = max(0.0, min(1.0, candidate.detection_score))
        weighted = (
            identity * self.settings.face_candidate_identity_weight
            + geometry * self.settings.face_candidate_geometry_weight
            + quality * self.settings.face_candidate_quality_weight
            + detection * self.settings.face_candidate_detection_weight
        )
        total_weight = (
            self.settings.face_candidate_identity_weight
            + self.settings.face_candidate_geometry_weight
            + self.settings.face_candidate_quality_weight
            + self.settings.face_candidate_detection_weight
        )
        if total_weight <= 0:
            raise ValueError("At least one face candidate score weight must be positive")
        return max(0.0, min(1.0, weighted / total_weight))

    def _geometry_score(self, candidate: FaceCandidate) -> float:
        if candidate.landmark_deformation_rms is None or candidate.landmark_deformation_max is None:
            return 0.0
        rms_limit = max(self.settings.landmark_deformation_rms_threshold, 1e-12)
        max_limit = max(self.settings.landmark_deformation_max_threshold, 1e-12)
        normalized_error = 0.5 * (
            candidate.landmark_deformation_rms / rms_limit
            + candidate.landmark_deformation_max / max_limit
        )
        return max(0.0, min(1.0, 1.0 - normalized_error))
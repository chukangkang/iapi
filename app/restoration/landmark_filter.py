import math
from dataclasses import dataclass, replace
from typing import Sequence

from app.config import Settings
from app.restoration.codeformer_service import FaceCandidateResult


Landmarks = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class LandmarkDeformationScore:
    rms: float
    maximum: float


def landmark_deformation_score(reference: Landmarks, candidate: Landmarks) -> LandmarkDeformationScore:
    """Measure non-rigid landmark error after the best similarity alignment."""
    if len(reference) != len(candidate) or len(reference) < 3:
        raise ValueError("Landmark sets must have the same length and at least three points")
    ref_center = _center(reference)
    candidate_center = _center(candidate)
    ref = tuple((x - ref_center[0], y - ref_center[1]) for x, y in reference)
    current = tuple((x - candidate_center[0], y - candidate_center[1]) for x, y in candidate)
    denominator = sum(x * x + y * y for x, y in current)
    reference_scale = math.sqrt(sum(x * x + y * y for x, y in ref) / len(ref))
    if denominator <= 1e-12 or reference_scale <= 1e-12:
        raise ValueError("Landmarks have degenerate geometry")

    scale_cos = sum(cx * rx + cy * ry for (cx, cy), (rx, ry) in zip(current, ref)) / denominator
    scale_sin = sum(cx * ry - cy * rx for (cx, cy), (rx, ry) in zip(current, ref)) / denominator
    errors = []
    for (cx, cy), (rx, ry) in zip(current, ref):
        aligned_x = scale_cos * cx - scale_sin * cy
        aligned_y = scale_sin * cx + scale_cos * cy
        errors.append(math.hypot(aligned_x - rx, aligned_y - ry) / reference_scale)
    return LandmarkDeformationScore(
        rms=math.sqrt(sum(error * error for error in errors) / len(errors)),
        maximum=max(errors),
    )


def _center(points: Landmarks) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


class LandmarkDeformationFilter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def filter_candidates(self, result: FaceCandidateResult) -> FaceCandidateResult:
        filtered = []
        for candidate in result.candidates:
            if not candidate.identity_accepted:
                filtered.append(
                    replace(
                        candidate,
                        landmark_deformation_rms=None,
                        landmark_deformation_max=None,
                        landmark_accepted=False,
                    )
                )
                continue
            try:
                score = landmark_deformation_score(
                    candidate.original_aligned_landmarks,
                    candidate.restored_aligned_landmarks,
                )
                accepted = (
                    score.rms <= self.settings.landmark_deformation_rms_threshold
                    and score.maximum <= self.settings.landmark_deformation_max_threshold
                )
                filtered.append(
                    replace(
                        candidate,
                        landmark_deformation_rms=score.rms,
                        landmark_deformation_max=score.maximum,
                        landmark_accepted=accepted,
                    )
                )
            except ValueError:
                filtered.append(
                    replace(
                        candidate,
                        landmark_deformation_rms=None,
                        landmark_deformation_max=None,
                        landmark_accepted=False,
                    )
                )
        return FaceCandidateResult(
            source_image=result.source_image,
            candidates=tuple(filtered),
            detected_face_count=result.detected_face_count,
        )
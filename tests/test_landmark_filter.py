from dataclasses import replace

import pytest
from PIL import Image

from app.config import Settings
from app.restoration.codeformer_service import FaceCandidate, FaceCandidateResult
from app.restoration.landmark_filter import LandmarkDeformationFilter, landmark_deformation_score


BASE_LANDMARKS = (
    (10.0, 10.0),
    (30.0, 10.0),
    (20.0, 20.0),
    (13.0, 30.0),
    (27.0, 30.0),
)


def _candidate(**changes) -> FaceCandidate:
    candidate = FaceCandidate(
        face_index=0,
        bbox=(0.0, 0.0, 40.0, 40.0),
        detection_score=0.9,
        original_face=Image.new("RGB", (40, 40), "gray"),
        restored_face=Image.new("RGB", (40, 40), "gray"),
        affine_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        landmarks=BASE_LANDMARKS,
        identity_score=0.9,
        identity_accepted=True,
        original_aligned_landmarks=BASE_LANDMARKS,
        restored_aligned_landmarks=BASE_LANDMARKS,
    )
    return replace(candidate, **changes)


def test_landmark_score_ignores_translation_rotation_and_uniform_scale():
    transformed = tuple((-2.0 * y + 100.0, 2.0 * x + 50.0) for x, y in BASE_LANDMARKS)

    score = landmark_deformation_score(BASE_LANDMARKS, transformed)

    assert score.rms == pytest.approx(0.0, abs=1e-7)
    assert score.maximum == pytest.approx(0.0, abs=1e-7)


def test_landmark_score_detects_local_feature_deformation():
    deformed = list(BASE_LANDMARKS)
    deformed[2] = (35.0, 20.0)

    score = landmark_deformation_score(BASE_LANDMARKS, tuple(deformed))

    assert score.rms > 0.1
    assert score.maximum > 0.2


def test_filter_accepts_identity_safe_rigid_candidate():
    service = LandmarkDeformationFilter(
        Settings(_env_file=None, landmark_deformation_rms_threshold=0.08, landmark_deformation_max_threshold=0.16)
    )
    result = FaceCandidateResult(Image.new("RGB", (40, 40)), (_candidate(),), 1)

    filtered = service.filter_candidates(result)

    assert filtered.candidates[0].landmark_accepted is True
    assert filtered.candidates[0].landmark_deformation_rms == pytest.approx(0.0)


def test_filter_rejects_local_feature_deformation():
    deformed = list(BASE_LANDMARKS)
    deformed[2] = (35.0, 20.0)
    service = LandmarkDeformationFilter(Settings(_env_file=None))
    result = FaceCandidateResult(
        Image.new("RGB", (40, 40)),
        (_candidate(restored_aligned_landmarks=tuple(deformed)),),
        1,
    )

    filtered = service.filter_candidates(result)

    assert filtered.candidates[0].landmark_accepted is False


def test_filter_rejects_missing_landmarks_and_identity_rejection():
    service = LandmarkDeformationFilter(Settings(_env_file=None))
    candidates = (
        _candidate(original_aligned_landmarks=()),
        _candidate(identity_accepted=False),
    )

    filtered = service.filter_candidates(FaceCandidateResult(Image.new("RGB", (40, 40)), candidates, 2))

    assert all(candidate.landmark_accepted is False for candidate in filtered.candidates)
from dataclasses import replace

import pytest
from PIL import Image

from app.config import Settings
from app.restoration.candidate_selector import FaceCandidateSelector, detail_score
from app.restoration.codeformer_service import FaceCandidate, FaceCandidateResult


def _candidate(**changes) -> FaceCandidate:
    candidate = FaceCandidate(
        face_index=0,
        bbox=(0.0, 0.0, 32.0, 32.0),
        detection_score=0.9,
        original_face=Image.new("RGB", (32, 32), "gray"),
        restored_face=_checkerboard(),
        affine_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        landmarks=(),
        identity_score=0.9,
        identity_accepted=True,
        landmark_deformation_rms=0.02,
        landmark_deformation_max=0.04,
        landmark_accepted=True,
    )
    return replace(candidate, **changes)


def _checkerboard() -> Image.Image:
    image = Image.new("RGB", (32, 32), "black")
    for y in range(32):
        for x in range(32):
            if (x + y) % 2:
                image.putpixel((x, y), (255, 255, 255))
    return image


def test_detail_score_rewards_restored_high_frequency_detail():
    assert detail_score(_checkerboard()) > detail_score(Image.new("RGB", (32, 32), "gray"))


def test_selector_computes_weighted_score_and_selects_safe_candidate():
    settings = Settings(
        _env_file=None,
        face_candidate_min_score=0.6,
        face_candidate_identity_weight=0.5,
        face_candidate_geometry_weight=0.3,
        face_candidate_quality_weight=0.15,
        face_candidate_detection_weight=0.05,
    )
    result = FaceCandidateResult(Image.new("RGB", (32, 32)), (_candidate(),), 1)

    selected = FaceCandidateSelector(settings).select(result)

    candidate = selected.candidates[0]
    assert candidate.quality_score > 0.5
    assert candidate.composite_score >= 0.6
    assert candidate.selected is True
    assert candidate.rejection_reason is None


def test_selector_rejects_candidate_that_failed_hard_gate():
    candidate = _candidate(identity_accepted=False)

    selected = FaceCandidateSelector(Settings(_env_file=None)).select(
        FaceCandidateResult(Image.new("RGB", (32, 32)), (candidate,), 1)
    )

    assert selected.candidates[0].selected is False
    assert selected.candidates[0].rejection_reason == "identity"


def test_selector_rejects_safe_but_low_composite_score():
    settings = Settings(_env_file=None, face_candidate_min_score=0.99)
    candidate = _candidate(identity_score=0.66, detection_score=0.1)

    selected = FaceCandidateSelector(settings).select(
        FaceCandidateResult(Image.new("RGB", (32, 32)), (candidate,), 1)
    )

    assert selected.candidates[0].selected is False
    assert selected.candidates[0].rejection_reason == "score"


def test_selector_falls_back_when_quality_scoring_fails(monkeypatch):
    selector = FaceCandidateSelector(Settings(_env_file=None))
    monkeypatch.setattr(selector, "_quality_score", lambda _candidate: (_ for _ in ()).throw(RuntimeError("bad")))

    selected = selector.select(
        FaceCandidateResult(Image.new("RGB", (32, 32)), (_candidate(),), 1)
    )

    assert selected.candidates[0].selected is False
    assert selected.candidates[0].rejection_reason == "scoring_error"
import pytest
from PIL import Image

from app.config import Settings
from app.restoration.codeformer_service import FaceCandidate, FaceCandidateResult
from app.restoration.identity_service import ArcFaceIdentityService, cosine_similarity


def _candidate(index: int, original_color: str = "black", restored_color: str = "white") -> FaceCandidate:
    return FaceCandidate(
        face_index=index,
        bbox=(0.0, 0.0, 16.0, 16.0),
        detection_score=0.9,
        original_face=Image.new("RGB", (16, 16), original_color),
        restored_face=Image.new("RGB", (16, 16), restored_color),
        affine_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        landmarks=(),
    )


def test_cosine_similarity_handles_parallel_and_opposite_embeddings():
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_rejects_zero_embedding():
    with pytest.raises(ValueError, match="zero-length"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


@pytest.mark.asyncio
async def test_disabled_identity_scoring_returns_unscored_candidates():
    result = FaceCandidateResult(Image.new("RGB", (32, 32)), (_candidate(0),), 1)
    service = ArcFaceIdentityService(Settings(_env_file=None, insightface_enabled=False))

    scored = await service.score_candidates(result)

    assert len(scored.candidates) == 1
    assert scored.candidates[0].identity_score is None
    assert scored.candidates[0].identity_accepted is False


def test_arcface_scores_and_applies_identity_threshold(monkeypatch):
    result = FaceCandidateResult(
        Image.new("RGB", (32, 32)),
        (_candidate(0, "red", "green"), _candidate(1, "blue", "yellow")),
        2,
    )
    service = ArcFaceIdentityService(
        Settings(_env_file=None, insightface_enabled=True, insightface_identity_threshold=0.65)
    )
    embeddings = {
        "red": [1.0, 0.0],
        "green": [0.8, 0.6],
        "blue": [1.0, 0.0],
        "yellow": [0.0, 1.0],
    }
    monkeypatch.setattr(service, "_get_analyzer", lambda: FakeAnalyzer(embeddings))
    monkeypatch.setattr(service, "_image_to_bgr", lambda image: image.getpixel((0, 0)))

    scored = service._score_sync(result)

    assert scored.candidates[0].identity_score == pytest.approx(0.8)
    assert scored.candidates[0].identity_accepted is True
    assert len(scored.candidates[0].original_aligned_landmarks) == 5
    assert len(scored.candidates[0].restored_aligned_landmarks) == 5
    assert scored.candidates[1].identity_score == pytest.approx(0.0)
    assert scored.candidates[1].identity_accepted is False


def test_arcface_isolates_candidate_without_detectable_embedding(monkeypatch):
    result = FaceCandidateResult(
        Image.new("RGB", (32, 32)),
        (_candidate(0, "red", "green"), _candidate(1, "blue", "yellow")),
        2,
    )
    service = ArcFaceIdentityService(Settings(_env_file=None, insightface_enabled=True))
    embeddings = {"red": [1.0, 0.0], "green": [1.0, 0.0], "blue": [1.0, 0.0]}
    monkeypatch.setattr(service, "_get_analyzer", lambda: FakeAnalyzer(embeddings))
    monkeypatch.setattr(service, "_image_to_bgr", lambda image: image.getpixel((0, 0)))

    scored = service._score_sync(result)

    assert scored.candidates[0].identity_accepted is True
    assert scored.candidates[1].identity_score is None
    assert scored.candidates[1].identity_accepted is False


class FakeFace:
    def __init__(self, embedding):
        self.normed_embedding = embedding
        self.bbox = (0, 0, 10, 10)
        self.kps = ((1, 1), (9, 1), (5, 5), (2, 9), (8, 9))


class FakeAnalyzer:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def get(self, color):
        names = {
            (255, 0, 0): "red",
            (0, 128, 0): "green",
            (0, 0, 255): "blue",
            (255, 255, 0): "yellow",
        }
        embedding = self.embeddings.get(names[color])
        return [] if embedding is None else [FakeFace(embedding)]
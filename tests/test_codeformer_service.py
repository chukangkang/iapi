from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.restoration.codeformer_service import CodeFormerService


def test_codeformer_default_path_uses_configured_checkpoint():
    service = CodeFormerService(Settings(_env_file=None))

    assert service.resolve_model_path() == Path("weights/CodeFormer/codeformer.pth")


def test_codeformer_missing_checkpoint_rejects_when_download_disabled(tmp_path):
    checkpoint = tmp_path / "codeformer.pth"
    service = CodeFormerService(
        Settings(
            _env_file=None,
            codeformer_model_path=str(checkpoint),
            codeformer_auto_download=False,
        )
    )

    with pytest.raises(FileNotFoundError, match="CodeFormer checkpoint"):
        service.ensure_model_path()


@pytest.mark.asyncio
async def test_disabled_codeformer_returns_no_candidates():
    image = Image.new("RGB", (32, 32), "gray")
    service = CodeFormerService(Settings(_env_file=None, codeformer_enabled=False))

    result = await service.generate_candidates(image)

    assert result.source_image is image
    assert result.candidates == ()


def test_codeformer_generates_one_candidate_per_detected_face(monkeypatch):
    helper = FakeFaceHelper()
    model = FakeCodeFormerModel()
    service = CodeFormerService(Settings(_env_file=None, codeformer_enabled=True))
    monkeypatch.setattr(service, "_resolve_device", lambda: FakeDevice())
    monkeypatch.setattr(service, "_image_to_bgr", lambda image: image)
    monkeypatch.setattr(service, "_create_face_helper", lambda _device: helper)
    monkeypatch.setattr(service, "_get_model", lambda _device: model)

    result = service._generate_sync(Image.new("RGB", (64, 64), "gray"))

    assert len(result.candidates) == 2
    assert result.candidates[0].bbox == (1.0, 2.0, 20.0, 21.0)
    assert result.candidates[0].original_face.size == (16, 16)
    assert result.candidates[0].restored_face.size == (16, 16)
    assert result.candidates[1].face_index == 1
    assert model.weights == [0.7, 0.7]


def test_codeformer_keeps_other_candidates_when_one_face_fails(monkeypatch):
    helper = FakeFaceHelper()
    model = FakeCodeFormerModel(fail_at=0)
    service = CodeFormerService(Settings(_env_file=None, codeformer_enabled=True))
    monkeypatch.setattr(service, "_resolve_device", lambda: FakeDevice())
    monkeypatch.setattr(service, "_image_to_bgr", lambda image: image)
    monkeypatch.setattr(service, "_create_face_helper", lambda _device: helper)
    monkeypatch.setattr(service, "_get_model", lambda _device: model)

    result = service._generate_sync(Image.new("RGB", (64, 64), "gray"))

    assert len(result.candidates) == 1
    assert result.candidates[0].face_index == 1


class FakeFaceHelper:
    def __init__(self):
        self.det_faces = [
            [1, 2, 20, 21, 0.9],
            [22, 3, 42, 23, 0.8],
        ]
        self.cropped_faces = [
            Image.new("RGB", (16, 16), (32, 32, 32)),
            Image.new("RGB", (16, 16), (64, 64, 64)),
        ]
        self.affine_matrices = [((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 1, 0))]

    def clean_all(self):
        pass

    def read_image(self, _image):
        pass

    def get_face_landmarks_5(self, **_kwargs):
        return 2

    def align_warp_face(self):
        pass


class FakeDevice:
    type = "cpu"


class FakeCodeFormerModel:
    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.weights = []

    def restore(self, face, *, fidelity_weight):
        index = len(self.weights)
        self.weights.append(fidelity_weight)
        if index == self.fail_at:
            raise RuntimeError("bad face")
        return Image.new("RGB", face.size, "white")
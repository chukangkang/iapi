from pathlib import Path

import pytest

from app.config import Settings
from app.upscale_service import ImageUpscaleService, _normalize_realesrgan_model_name


def test_realesrgan_default_path_uses_weights_directory():
    service = ImageUpscaleService(Settings(realesrgan_model_path="", realesrgan_model_name="realesr-general-x4v3"))

    model_path, spec = service._resolve_realesrgan_model()

    assert model_path == Path("weights") / "realesr-general-x4v3.pth"
    assert spec.name == "realesr-general-x4v3"


def test_realesrgan_directory_path_appends_official_checkpoint_name(tmp_path):
    model_dir = tmp_path / "models"
    service = ImageUpscaleService(Settings(realesrgan_model_path=str(model_dir), realesrgan_model_name="RealESRGAN_x4plus.pth"))

    model_path, spec = service._resolve_realesrgan_model()

    assert model_path == model_dir / "RealESRGAN_x4plus.pth"
    assert spec.name == "RealESRGAN_x4plus"


def test_realesrgan_rejects_safetensors_checkpoint(tmp_path):
    service = ImageUpscaleService(Settings(realesrgan_model_path=str(tmp_path / "model.safetensors")))

    with pytest.raises(ValueError, match=".pth checkpoint"):
        service._resolve_realesrgan_model()


def test_realesrgan_model_name_aliases_match_official_ncnn_names():
    assert _normalize_realesrgan_model_name("realesrgan-x4plus") == "realesrgan_x4plus"
    assert _normalize_realesrgan_model_name("realesrgan-x4plus-anime") == "realesrgan_x4plus_anime_6b"


def test_realesrgan_general_model_uses_denoise_pair_when_enabled(tmp_path):
    model_path = tmp_path / "realesr-general-x4v3.pth"
    wdn_path = tmp_path / "realesr-general-wdn-x4v3.pth"
    model_path.write_bytes(b"model")
    wdn_path.write_bytes(b"wdn")
    service = ImageUpscaleService(
        Settings(
            realesrgan_model_path=str(model_path),
            realesrgan_model_name="realesr-general-x4v3",
            realesrgan_denoise_strength=0.35,
        )
    )
    _, spec = service._resolve_realesrgan_model()

    _, model_paths = service._ensure_model_paths(model_path, spec, _unexpected_download)

    assert model_paths == [str(model_path), str(wdn_path)]


def test_realesrgan_outscale_zero_uses_model_scale():
    service = ImageUpscaleService(Settings(realesrgan_outscale=0, realesrgan_model_name="RealESRGAN_x2plus"))
    spec = service._resolve_realesrgan_model_spec()

    assert service._resolve_outscale(spec) == 2.0


def test_realesrgan_outscale_env_overrides_model_scale():
    service = ImageUpscaleService(Settings(realesrgan_outscale=3.5, realesrgan_model_name="RealESRGAN_x4plus"))
    spec = service._resolve_realesrgan_model_spec()

    assert service._resolve_outscale(spec) == 3.5


def test_realesrgan_upsampler_uses_env_defaults(tmp_path):
    model_path = tmp_path / "RealESRGAN_x4plus.pth"
    model_path.write_bytes(b"model")
    service = ImageUpscaleService(
        Settings(
            realesrgan_model_path=str(model_path),
            realesrgan_model_name="RealESRGAN_x4plus",
            realesrgan_tile=128,
            realesrgan_tile_pad=12,
            realesrgan_pre_pad=3,
            realesrgan_fp32=True,
            realesrgan_gpu_id=1,
        )
    )
    _, spec = service._resolve_realesrgan_model()

    service._get_upsampler(FakeRealESRGANer, FakeRRDBNet, FakeSRVGGNet, _unexpected_download, model_path, spec)

    assert FakeRealESRGANer.last_kwargs["tile"] == 128
    assert FakeRealESRGANer.last_kwargs["tile_pad"] == 12
    assert FakeRealESRGANer.last_kwargs["pre_pad"] == 3
    assert FakeRealESRGANer.last_kwargs["half"] is False
    assert FakeRealESRGANer.last_kwargs["gpu_id"] == 1


def test_realesrgan_face_enhancer_uses_official_gfpgan_defaults(tmp_path):
    model_path = tmp_path / "RealESRGAN_x4plus.pth"
    model_path.write_bytes(b"model")
    service = ImageUpscaleService(
        Settings(
            realesrgan_model_path=str(model_path),
            realesrgan_model_name="RealESRGAN_x4plus",
            realesrgan_face_enhance=True,
            realesrgan_outscale=3,
        )
    )
    _, spec = service._resolve_realesrgan_model()
    upsampler = service._get_upsampler(FakeRealESRGANer, FakeRRDBNet, FakeSRVGGNet, _unexpected_download, model_path, spec)

    service._get_face_enhancer(FakeGFPGANer, upsampler)

    assert FakeGFPGANer.last_kwargs["model_path"].endswith("GFPGANv1.3.pth")
    assert FakeGFPGANer.last_kwargs["upscale"] == 3
    assert FakeGFPGANer.last_kwargs["arch"] == "clean"
    assert FakeGFPGANer.last_kwargs["channel_multiplier"] == 2
    assert FakeGFPGANer.last_kwargs["bg_upsampler"] is upsampler


def _unexpected_download(**_kwargs):
    raise AssertionError("download should not be called")


class FakeRealESRGANer:
    last_kwargs = None

    def __init__(self, *, gpu_id=None, dni_weight=None, **kwargs):
        self.__class__.last_kwargs = {**kwargs, "gpu_id": gpu_id, "dni_weight": dni_weight}


class FakeRRDBNet:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeSRVGGNet:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeGFPGANer:
    last_kwargs = None

    def __init__(self, **kwargs):
        self.__class__.last_kwargs = kwargs
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.restoration.analyzer import DegradationReport
from app.restoration.swinir_service import SwinIRService


def _report(*, noise: float = 0.0, blockiness: float = 0.0) -> DegradationReport:
    return DegradationReport(
        blur_score=0.5,
        detail_score=0.5,
        noise_score=noise,
        blockiness_score=blockiness,
        exposure_score=0.0,
        recommended_mode="balanced",
    )


def test_swinir_selects_jpeg_model_for_block_artifacts():
    service = SwinIRService(Settings(_env_file=None))

    spec = service.select_model(_report(noise=0.2, blockiness=0.5))

    assert spec.task == "jpeg"
    assert spec.window_size == 7
    assert "jpeg40" in spec.filename


def test_swinir_selects_denoise_model_for_noise():
    service = SwinIRService(Settings(_env_file=None))

    spec = service.select_model(_report(noise=0.5, blockiness=0.1))

    assert spec.task == "denoise"
    assert spec.window_size == 8
    assert "noise50" in spec.filename


def test_swinir_explicit_checkpoint_disables_download(tmp_path):
    checkpoint = tmp_path / "custom-swinir.pth"
    checkpoint.write_bytes(b"checkpoint")
    service = SwinIRService(Settings(_env_file=None, swinir_model_path=str(checkpoint)))

    path = service.resolve_model_path(service.select_model(_report(noise=0.3)))

    assert path == checkpoint


def test_swinir_missing_explicit_checkpoint_is_not_silently_downloaded(tmp_path):
    checkpoint = tmp_path / "missing.pth"
    service = SwinIRService(Settings(_env_file=None, swinir_model_path=str(checkpoint)))

    with pytest.raises(FileNotFoundError, match="SWINIR_MODEL_PATH"):
        service.resolve_model_path(service.select_model(_report(noise=0.3)))


@pytest.mark.asyncio
async def test_disabled_swinir_returns_original_image():
    image = Image.new("RGB", (17, 19), "gray")
    service = SwinIRService(Settings(_env_file=None, swinir_enabled=False))

    restored = await service.restore(image, report=_report(noise=0.5))

    assert restored is image
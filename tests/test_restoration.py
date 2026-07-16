from PIL import Image, ImageFilter

from app.config import Settings
from app.restoration.analyzer import DegradationAnalyzer
from app.restoration.orchestrator import RestorationOrchestrator


def test_degradation_analyzer_detects_flat_blurry_image():
    image = Image.new("RGB", (128, 128), (128, 128, 128))

    report = DegradationAnalyzer().analyze(image)

    assert report.blur_score >= 0.9
    assert report.detail_score <= 0.1
    assert report.recommended_mode == "balanced"


def test_degradation_analyzer_detects_high_detail_image():
    image = Image.new("RGB", (128, 128), "black")
    for x in range(0, 128, 4):
        for y in range(128):
            image.putpixel((x, y), (255, 255, 255))

    report = DegradationAnalyzer().analyze(image)

    assert report.detail_score > 0.2
    assert report.recommended_mode == "preserve"


def test_preserve_mode_uses_conservative_realesrnet():
    settings = Settings(_env_file=None)
    orchestrator = RestorationOrchestrator(settings)

    plan = orchestrator.plan("preserve", Image.new("RGB", (64, 64), "gray"))

    assert plan.use_qwen_edit is False
    assert plan.upscale_method == "realesrgan"
    assert plan.realesrgan_model_name == "RealESRNet_x4plus"
    assert plan.face_restoration is False


def test_balanced_mode_does_not_use_qwen_edit_by_default():
    settings = Settings(_env_file=None, swinir_enabled=True)
    orchestrator = RestorationOrchestrator(settings)
    blurry = Image.new("RGB", (64, 64), "gray").filter(ImageFilter.GaussianBlur(4))

    plan = orchestrator.plan("balanced", blurry)

    assert plan.use_qwen_edit is False
    assert plan.use_swinir is True
    assert plan.upscale_method == "realesrgan"
    assert plan.realesrgan_model_name == settings.restoration_balanced_realesrgan_model_name


def test_creative_mode_uses_qwen_edit_and_can_route_to_supir():
    settings = Settings(_env_file=None, supir_enabled=True, supir_base_url="http://supir:8000")
    orchestrator = RestorationOrchestrator(settings)

    plan = orchestrator.plan("creative", Image.new("RGB", (64, 64), "gray"))

    assert plan.use_qwen_edit is True
    assert plan.use_supir is True


def test_preserve_mode_does_not_route_to_swinir():
    settings = Settings(_env_file=None, swinir_enabled=True)
    orchestrator = RestorationOrchestrator(settings)

    plan = orchestrator.plan("preserve", Image.new("RGB", (64, 64), "gray"))

    assert plan.use_swinir is False


def test_auto_mode_uses_analyzer_recommendation():
    settings = Settings(_env_file=None)
    orchestrator = RestorationOrchestrator(settings)

    plan = orchestrator.plan("auto", Image.new("RGB", (64, 64), "gray"))

    assert plan.mode == "balanced"

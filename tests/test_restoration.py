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


def test_auto_mode_routes_anime_image_to_anime_realesrgan():
    settings = Settings(
        _env_file=None,
        codeformer_enabled=True,
        swinir_enabled=True,
        restoration_anime_detection_enabled=True,
        restoration_anime_score_threshold=0.45,
    )
    image = Image.new("RGB", (128, 128), (245, 235, 210))
    for x in range(16, 112):
        image.putpixel((x, 24), (20, 20, 30))
        image.putpixel((x, 104), (20, 20, 30))
    for y in range(24, 105):
        image.putpixel((16, y), (20, 20, 30))
        image.putpixel((111, y), (20, 20, 30))
    for y in range(25, 104):
        for x in range(17, 111):
            image.putpixel((x, y), (90, 70, 150) if x < 64 else (190, 165, 110))

    plan = RestorationOrchestrator(settings).plan("auto", image)

    assert plan.report.is_anime is True
    assert plan.realesrgan_model_name == "RealESRGAN_x4plus_anime_6B"
    assert plan.face_restoration is False


def test_explicit_preserve_mode_does_not_force_anime_model():
    settings = Settings(_env_file=None, restoration_anime_detection_enabled=True)
    image = Image.new("RGB", (64, 64), (220, 180, 100))

    plan = RestorationOrchestrator(settings).plan("preserve", image)

    assert plan.realesrgan_model_name == settings.restoration_preserve_realesrgan_model_name

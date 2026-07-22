import math

import pytest
from PIL import Image, ImageFilter

from app.config import Settings
from app.restoration.analyzer import DegradationAnalyzer
from app.restoration.orchestrator import RestorationOrchestrator
from app.restoration.style_classifier import IllustrationClassification, IllustrationStyleClassifier


@pytest.fixture(autouse=True)
def _disable_style_classifier_by_default(monkeypatch):
    monkeypatch.setenv("RESTORATION_STYLE_CLASSIFIER_ENABLED", "false")


def _sparse_clear_portrait() -> Image.Image:
    image = Image.new("RGB", (256, 256))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            luminance = int(205 + 14 * y / image.height)
            pixels[x, y] = (luminance - 18, luminance - 5, min(255, luminance + 8))

    for y in range(36, 175):
        for x in range(78, 179):
            normalized_x = (x - 128) / 50
            normalized_y = (y - 105) / 69
            if normalized_x**2 + normalized_y**2 <= 1:
                shade = int(205 - 18 * (normalized_x**2 + normalized_y**2))
                pixels[x, y] = (shade, shade - 50, shade - 80)
    for x in range(70, 186):
        image.putpixel((x, 65), (30, 28, 29))
        image.putpixel((x, 66), (30, 28, 29))
    for x in range(105, 122):
        image.putpixel((x, 91), (55, 42, 39))
    for x in range(139, 156):
        image.putpixel((x, 91), (55, 42, 39))
    for y in range(162, image.height):
        left = max(0, 73 - (y - 162) // 2)
        right = min(image.width, 183 + (y - 162) // 2)
        for x in range(left, right):
            pixels[x, y] = (195, 186, 176)
    return image


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


def test_auto_mode_preserves_sparse_clear_portrait_without_denoising():
    settings = Settings(_env_file=None, swinir_enabled=True)

    plan = RestorationOrchestrator(settings).plan("auto", _sparse_clear_portrait())

    assert plan.report.recommended_mode == "preserve"
    assert plan.mode == "preserve"
    assert plan.realesrgan_model_name == settings.restoration_preserve_realesrgan_model_name
    assert plan.use_swinir is False
    assert plan.use_qwen_edit is False


def test_auto_mode_routes_genuinely_blurred_portrait_without_unneeded_denoising():
    settings = Settings(
        _env_file=None,
        swinir_enabled=True,
        restoration_severe_blur_enabled=False,
    )
    blurry = _sparse_clear_portrait().filter(ImageFilter.GaussianBlur(3))

    plan = RestorationOrchestrator(settings).plan("auto", blurry)

    assert plan.report.recommended_mode == "balanced"
    assert plan.mode == "balanced"
    assert plan.realesrgan_model_name == settings.restoration_balanced_realesrgan_model_name
    assert plan.use_swinir is False


def test_auto_mode_enables_swinir_for_detected_noise():
    settings = Settings(
        _env_file=None,
        swinir_enabled=True,
        restoration_severe_blur_enabled=False,
    )
    noisy = Image.new("RGB", (128, 128))
    pixels = noisy.load()
    for y in range(noisy.height):
        for x in range(noisy.width):
            value = (x * 73 + y * 151 + x * y * 19) % 256
            pixels[x, y] = (value, value, value)

    plan = RestorationOrchestrator(settings).plan("auto", noisy)

    assert plan.report.noise_score >= 0.45
    assert plan.mode == "balanced"
    assert plan.use_swinir is True


def test_auto_mode_does_not_escalate_moderate_blur_to_generative_restoration():
    settings = Settings(
        _env_file=None,
        swinir_enabled=True,
        restoration_severe_blur_enabled=True,
        restoration_severe_blur_threshold=0.82,
    )
    moderately_blurry = _sparse_clear_portrait().filter(ImageFilter.GaussianBlur(1.5))

    plan = RestorationOrchestrator(settings).plan("auto", moderately_blurry)

    assert plan.mode == "balanced"
    assert plan.severe_blur is False
    assert plan.use_qwen_edit is False
    assert plan.use_supir is False


def test_auto_mode_escalates_severe_photo_blur_to_qwen_hd_restoration():
    settings = Settings(
        _env_file=None,
        restoration_severe_blur_enabled=True,
        restoration_severe_blur_threshold=0.8,
        supir_enabled=False,
    )

    plan = RestorationOrchestrator(settings).plan("auto", Image.new("RGB", (96, 96), "gray"))

    assert plan.severe_blur is True
    assert plan.use_qwen_edit is True
    assert plan.use_supir is False
    assert plan.realesrgan_model_name == settings.restoration_creative_realesrgan_model_name


def test_auto_mode_prefers_supir_for_severe_blur_when_available():
    settings = Settings(
        _env_file=None,
        restoration_severe_blur_enabled=True,
        restoration_severe_blur_threshold=0.8,
        restoration_severe_blur_prefer_supir=True,
        supir_enabled=True,
        supir_base_url="http://supir-worker:8010",
    )

    plan = RestorationOrchestrator(settings).plan("auto", Image.new("RGB", (96, 96), "gray"))

    assert plan.severe_blur is True
    assert plan.use_supir is True
    assert plan.use_qwen_edit is False


def test_auto_mode_can_disable_severe_blur_escalation():
    settings = Settings(
        _env_file=None,
        restoration_severe_blur_enabled=False,
        restoration_severe_blur_threshold=0.8,
    )

    plan = RestorationOrchestrator(settings).plan("auto", Image.new("RGB", (96, 96), "gray"))

    assert plan.severe_blur is False
    assert plan.use_qwen_edit is False


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


def test_auto_mode_routes_semantic_game_illustration_to_anime_realesrgan():
    settings = Settings(
        _env_file=None,
        restoration_anime_detection_enabled=True,
        restoration_style_classifier_enabled=True,
        restoration_style_classifier_threshold=0.72,
    )
    image = Image.new("RGB", (512, 256), (196, 188, 168))
    classifier = FakeStyleClassifier(
        IllustrationClassification(
            illustration_score=0.96,
            photo_score=0.04,
            label="illustrated promotional poster",
        )
    )

    plan = RestorationOrchestrator(settings, style_classifier=classifier).plan("auto", image)

    assert classifier.images == [image]
    assert plan.report.is_anime is False
    assert plan.is_illustration is True
    assert plan.illustration_score == pytest.approx(0.96)
    assert plan.realesrgan_model_name == settings.restoration_anime_realesrgan_model_name
    assert plan.use_swinir is False
    assert plan.face_restoration is False


def test_auto_mode_keeps_photo_route_when_semantic_style_classifier_fails():
    settings = Settings(
        _env_file=None,
        restoration_anime_detection_enabled=True,
        restoration_style_classifier_enabled=True,
    )
    classifier = FakeStyleClassifier(error=RuntimeError("classifier unavailable"))

    plan = RestorationOrchestrator(settings, style_classifier=classifier).plan(
        "auto",
        _sparse_clear_portrait(),
    )

    assert plan.is_illustration is False
    assert plan.realesrgan_model_name == settings.restoration_preserve_realesrgan_model_name


def test_auto_mode_does_not_route_continuous_tone_photo_to_anime_model():
    settings = Settings(
        _env_file=None,
        restoration_anime_detection_enabled=True,
        restoration_anime_score_threshold=0.68,
    )
    image = Image.new("RGB", (160, 240))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            luminance = int(
                120
                + 35 * math.sin(x / 9)
                + 25 * math.sin(y / 13)
                + ((x * 37 + y * 61) % 17)
                - 8
            )
            luminance = max(0, min(255, luminance))
            pixels[x, y] = (luminance, max(0, luminance - 8), max(0, luminance - 14))
    for x in range(0, image.width, 16):
        for y in range(image.height):
            image.putpixel((x, y), (30, 35, 40))
    for y in range(0, image.height, 18):
        for x in range(image.width):
            image.putpixel((x, y), (45, 48, 52))

    plan = RestorationOrchestrator(settings).plan("auto", image)

    assert plan.report.is_anime is False
    assert plan.realesrgan_model_name == settings.restoration_preserve_realesrgan_model_name


def test_style_classifier_balances_photo_and_illustration_scores():
    classifier = IllustrationStyleClassifier(
        Settings(_env_file=None, restoration_style_classifier_enabled=True)
    )
    classifier._classifier = FakeZeroShotPipeline(
        [
            {"label": IllustrationStyleClassifier.PHOTO_LABEL, "score": 0.20},
            {"label": IllustrationStyleClassifier.ILLUSTRATION_LABEL, "score": 0.80},
        ]
    )

    result = classifier.classify(Image.new("RGB", (64, 64), "gray"))

    assert result.photo_score == pytest.approx(0.20)
    assert result.illustration_score == pytest.approx(0.80)
    assert result.label == IllustrationStyleClassifier.ILLUSTRATION_LABEL


def test_style_classifier_counts_photographic_advertising_as_photo():
    classifier = IllustrationStyleClassifier(
        Settings(_env_file=None, restoration_style_classifier_enabled=True)
    )
    classifier._classifier = FakeZeroShotPipeline(
        [
            {"label": IllustrationStyleClassifier.PHOTO_LABEL, "score": 0.86},
            {"label": IllustrationStyleClassifier.ILLUSTRATION_LABEL, "score": 0.14},
        ]
    )

    result = classifier.classify(Image.new("RGB", (64, 64), "gray"))

    assert result.photo_score == pytest.approx(0.86)
    assert result.illustration_score == pytest.approx(0.14)
    assert result.label == IllustrationStyleClassifier.PHOTO_LABEL


def test_auto_mode_prioritizes_severe_photo_blur_after_photo_classification():
    settings = Settings(
        _env_file=None,
        restoration_anime_detection_enabled=True,
        restoration_style_classifier_enabled=True,
        restoration_style_classifier_threshold=0.72,
        restoration_severe_blur_enabled=True,
        restoration_severe_blur_threshold=0.80,
        restoration_severe_blur_use_qwen_edit=True,
        supir_enabled=False,
    )
    classifier = FakeStyleClassifier(
        IllustrationClassification(
            illustration_score=0.03,
            photo_score=0.97,
            label=IllustrationStyleClassifier.PHOTO_LABEL,
        )
    )

    plan = RestorationOrchestrator(settings, style_classifier=classifier).plan(
        "auto",
        Image.new("RGB", (96, 96), "gray"),
    )

    assert plan.is_illustration is False
    assert plan.severe_blur is True
    assert plan.use_qwen_edit is True
    assert plan.use_qwen_unblur_lora is settings.qwen_unblur_upscale_lora_enabled
    assert plan.realesrgan_model_name == settings.restoration_creative_realesrgan_model_name


def test_auto_mode_keeps_photographic_advertisement_on_photo_model():
    settings = Settings(
        _env_file=None,
        restoration_anime_detection_enabled=True,
        restoration_style_classifier_enabled=True,
        restoration_style_classifier_threshold=0.72,
    )
    classifier = FakeStyleClassifier(
        IllustrationClassification(
            illustration_score=0.14,
            photo_score=0.86,
            label=IllustrationStyleClassifier.PHOTO_LABEL,
        )
    )

    plan = RestorationOrchestrator(settings, style_classifier=classifier).plan(
        "auto",
        _sparse_clear_portrait(),
    )

    assert plan.is_illustration is False
    assert plan.photo_score == pytest.approx(0.86)
    assert plan.illustration_score == pytest.approx(0.14)
    assert plan.style_label == IllustrationStyleClassifier.PHOTO_LABEL
    assert plan.realesrgan_model_name == settings.restoration_preserve_realesrgan_model_name
    assert plan.use_qwen_edit is False


class FakeStyleClassifier:
    def __init__(self, result=None, *, error=None):
        self.result = result
        self.error = error
        self.images = []

    def classify(self, image):
        self.images.append(image)
        if self.error is not None:
            raise self.error
        return self.result


class FakeZeroShotPipeline:
    def __init__(self, result):
        self.result = result

    def __call__(self, _image, *, candidate_labels, hypothesis_template):
        assert candidate_labels == list(IllustrationStyleClassifier.CANDIDATE_LABELS)
        assert hypothesis_template == "{}"
        return self.result


def test_explicit_preserve_mode_does_not_force_anime_model():
    settings = Settings(_env_file=None, restoration_anime_detection_enabled=True)
    image = Image.new("RGB", (64, 64), (220, 180, 100))

    plan = RestorationOrchestrator(settings).plan("preserve", image)

    assert plan.realesrgan_model_name == settings.restoration_preserve_realesrgan_model_name

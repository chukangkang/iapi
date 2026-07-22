from PIL import Image

from app.config import Settings
from app.restoration.detail_blender import RestorationDetailBlender


def _striped_image(base: int, stripe: int) -> Image.Image:
    image = Image.new("RGB", (64, 64), (base, base, base))
    for x in range(0, 64, 4):
        for y in range(64):
            image.putpixel((x, y), (stripe, stripe, stripe))
    return image


def test_detail_blender_anchors_generated_low_frequency_tone(monkeypatch):
    settings = Settings(
        _env_file=None,
        restoration_generative_blend_enabled=True,
        restoration_generative_blend_strength=0.75,
        restoration_generative_blend_low_frequency_radius=6.0,
    )
    blender = RestorationDetailBlender(settings)
    monkeypatch.setattr(blender, "_face_mask", lambda _image: None)
    source = Image.new("RGB", (64, 64), (80, 80, 80))
    restored = Image.new("RGB", (64, 64), (200, 200, 200))

    result = blender.blend(source, restored)

    value = result.getpixel((32, 32))[0]
    assert 76 <= value <= 84


def test_detail_blender_keeps_generated_high_frequency_detail(monkeypatch):
    settings = Settings(
        _env_file=None,
        restoration_generative_blend_enabled=True,
        restoration_generative_blend_strength=0.8,
        restoration_generative_blend_low_frequency_radius=4.0,
    )
    blender = RestorationDetailBlender(settings)
    monkeypatch.setattr(blender, "_face_mask", lambda _image: None)
    source = Image.new("RGB", (64, 64), (100, 100, 100))
    restored = _striped_image(180, 240)

    result = blender.blend(source, restored)

    assert result.getpixel((0, 32))[0] > result.getpixel((2, 32))[0]
    assert result.getpixel((2, 32))[0] < 140


def test_detail_blender_can_be_disabled():
    settings = Settings(_env_file=None, restoration_generative_blend_enabled=False)
    restored = Image.new("RGB", (32, 32), "red")

    result = RestorationDetailBlender(settings).blend(Image.new("RGB", (32, 32), "blue"), restored)

    assert result.tobytes() == restored.tobytes()
import pytest
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from app.image_utils import image_to_base64_png, string_list_to_images, string_to_image
from app.main import ImageGenerationRequest, _apply_prompt_params, _payload_image_to_reference, _resolve_dimensions, _resolve_enhance_mode, _resolve_face_enhance, _resolve_negative_prompt, _resolve_qwen_edit_dimensions, _validate_image_payload
from app.config import Settings
from app.qwen_edit_service import QwenImageEditService
from app.upscale_service import ImageUpscaleService


def _sample_base64_png() -> str:
    image = Image.new("RGB", (2, 2), "red")
    return image_to_base64_png(image)


def test_string_to_image_accepts_raw_base64_with_whitespace():
    encoded = _sample_base64_png()
    wrapped = f"{encoded[:12]}\n {encoded[12:]}"

    image = string_to_image(wrapped)

    assert image is not None
    assert image.size == (2, 2)


def test_string_to_image_accepts_data_url_with_whitespace():
    encoded = _sample_base64_png()
    data_url = f"data:image/png;base64,{encoded[:10]}\n{encoded[10:]}"

    image = string_to_image(data_url)

    assert image is not None
    assert image.size == (2, 2)


def test_string_list_to_images_accepts_two_images():
    encoded = _sample_base64_png()

    images = string_list_to_images([encoded, encoded])

    assert len(images) == 2
    assert images[0].size == (2, 2)


def test_response_format_accepts_base64_alias():
    payload = ImageGenerationRequest(prompt="test", response_format="base64")

    _validate_image_payload(payload, Settings())

    assert payload.response_format == "b64_json"


def test_response_format_rejects_unknown_values():
    payload = ImageGenerationRequest(prompt="test", response_format="json")

    try:
        _validate_image_payload(payload, Settings())
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "response_format" in exc.detail
    else:
        raise AssertionError("Expected invalid response_format to fail")


def test_seed_defaults_to_none():
    payload = ImageGenerationRequest(prompt="test")

    assert payload.seed is None


def test_qwen_image_2512_uses_official_defaults():
    settings = Settings(_env_file=None)

    assert settings.qwen_image_model_path == "Qwen/Qwen-Image-2512"
    assert settings.qwen_image_steps == 50
    assert settings.qwen_image_true_cfg_scale == 4.0
    assert settings.qwen_image_negative_prompt == (
        "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
        "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
    )


def test_qwen_image_uses_fixed_negative_prompt_from_settings():
    settings = Settings(_env_file=None, qwen_image_negative_prompt="  固定中文负面提示词  ")

    assert _resolve_negative_prompt(settings) == "固定中文负面提示词"


def test_qwen_edit_scale_to_length_zero_disables_scaling():
    settings = Settings(_env_file=None, qwen_edit_scale_to_length=0, qwen_edit_round_to_multiple=16)

    assert _resolve_qwen_edit_dimensions(1025, 769, settings) == (1024, 768)


def test_qwen_edit_scale_to_length_rejects_negative_values():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qwen_edit_scale_to_length=-1)


def test_qwen_edit_cover_fit_preserves_full_image_content():
    settings = Settings(_env_file=None, qwen_edit_input_fit_mode="cover")
    service = QwenImageEditService(settings)
    image = Image.new("RGB", (200, 100), "red")

    prepared = service._prepare_image(image, 100, 100)

    assert prepared.size == (100, 100)
    assert prepared.getbbox() == (0, 25, 100, 75)


def test_upscale_cover_fit_preserves_full_image_content():
    settings = Settings(_env_file=None, upscale_fit_mode="cover", upscale_fill_color="black")
    service = ImageUpscaleService(settings)
    image = Image.new("RGB", (200, 100), "red")

    fitted = service._fit_to_target(image, width=100, height=100, fit_mode="cover")

    assert fitted.size == (100, 100)
    assert fitted.getbbox() == (0, 25, 100, 75)


def test_realesrgan_enhance_mode_allows_empty_model_path_for_default_weights():
    payload = ImageGenerationRequest(prompt="test", image=_sample_base64_png(), enhance_mode="realesrgan")

    _validate_image_payload(payload, Settings(service_role="api", realesrgan_model_path=""))


def test_image_request_accepts_two_image_values():
    encoded = _sample_base64_png()
    payload = ImageGenerationRequest(prompt="merge", model="qwen-image-2512", image=[encoded, encoded])

    _validate_image_payload(payload, Settings())

    reference_image = _payload_image_to_reference(payload.image)
    assert isinstance(reference_image, list)
    assert len(reference_image) == 2
    assert _resolve_enhance_mode(payload, Settings()) == "qwen_edit"


def test_image_request_rejects_three_image_values():
    encoded = _sample_base64_png()
    payload = ImageGenerationRequest(prompt="merge", image=[encoded, encoded, encoded])

    try:
        _validate_image_payload(payload, Settings())
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "one or two" in exc.detail
    else:
        raise AssertionError("Expected more than two image values to fail")


def test_generation_dimensions_use_reference_aspect_ratio_when_missing():
    payload = ImageGenerationRequest(prompt="test", image=_sample_base64_png())
    reference_image = Image.new("RGB", (1600, 900), "red")

    assert _resolve_dimensions(payload, Settings(), reference_image) == (1664, 928)


def test_edit_dimensions_use_reference_aspect_ratio_when_missing():
    payload = ImageGenerationRequest(endpoint="edits", prompt="test", image=_sample_base64_png())
    reference_image = Image.new("RGB", (900, 1600), "red")

    width, height = _resolve_dimensions(payload, Settings(), reference_image)

    assert height > width
    assert width % 16 == 0
    assert height % 16 == 0
    assert abs((width / height) - (9 / 16)) < 0.02


def test_edit_resolution_uses_reference_aspect_ratio_when_missing():
    payload = ImageGenerationRequest(endpoint="edits", prompt="test", image=_sample_base64_png(), resolution="2k")
    reference_image = Image.new("RGB", (1600, 900), "red")

    assert _resolve_dimensions(payload, Settings(), reference_image) == (2560, 1440)


def test_edit_4k_resolution_preserves_reference_aspect_ratio_when_missing():
    payload = ImageGenerationRequest(endpoint="edits", prompt="test", image=_sample_base64_png(), resolution="4k")
    reference_image = Image.new("RGB", (1000, 1500), "red")

    width, height = _resolve_dimensions(payload, Settings(), reference_image)

    assert (width, height) == (2736, 4096)
    assert abs((width / height) - (1000 / 1500)) < 0.002


def test_explicit_edit_dimensions_override_reference_aspect_ratio():
    payload = ImageGenerationRequest(endpoint="edits", prompt="test", image=_sample_base64_png(), size="1024x768")
    reference_image = Image.new("RGB", (900, 1600), "red")

    assert _resolve_dimensions(payload, Settings(), reference_image) == (1024, 768)


def test_face_enhance_request_overrides_settings():
    assert _resolve_face_enhance(ImageGenerationRequest(prompt="test", face_enhance=True), Settings(realesrgan_face_enhance=False)) is True
    assert _resolve_face_enhance(ImageGenerationRequest(prompt="test", face_enhance=False), Settings(realesrgan_face_enhance=True)) is False


def test_face_enhance_can_be_read_from_prompt_params():
    payload = ImageGenerationRequest(prompt="restore portrait [face_enhance=true]")

    _apply_prompt_params(payload)

    assert payload.face_enhance is True
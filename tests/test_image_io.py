from fastapi import HTTPException
from PIL import Image

from app.image_utils import image_to_base64_png, string_to_image
from app.main import ImageGenerationRequest, _validate_image_payload
from app.config import Settings


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
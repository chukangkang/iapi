import base64
import io
import re
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageOps
from fastapi import HTTPException, UploadFile, status

DATA_URL_PATTERN = re.compile(r"^data:image/[^;]+;base64,(?P<data>.+)$", re.IGNORECASE | re.DOTALL)


def image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def save_png(image: Image.Image, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    image.save(path, format="PNG")
    return path


def public_image_url(base_url: str, filename: str) -> str:
    return f"{base_url}/outputs/{filename}" if base_url else f"/outputs/{filename}"


async def upload_file_to_image(file: Optional[UploadFile]) -> Optional[Image.Image]:
    if file is None:
        return None

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded image is empty")
    return bytes_to_image(content)


def string_to_image(value: Optional[str]) -> Optional[Image.Image]:
    if not value:
        return None

    value = value.strip()
    data_url_match = DATA_URL_PATTERN.match(value)
    if data_url_match:
        return bytes_to_image(base64.b64decode(data_url_match.group("data"), validate=True))

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        request = Request(value, headers={"User-Agent": "iapi-openai-image-server/1.0"})
        try:
            with urlopen(request, timeout=30) as response:
                return bytes_to_image(response.read())
        except URLError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download image URL: {exc.reason}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download image URL: {exc}",
            ) from exc

    try:
        return bytes_to_image(base64.b64decode(value, validate=True))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image must be a URL, base64 string, data URL, or multipart file",
        ) from exc


def bytes_to_image(content: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(content))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image data") from exc

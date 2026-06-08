import time
import uuid
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.flux_service import FluxImageService
from app.image_utils import image_to_base64_png, public_image_url, save_png, string_to_image, upload_file_to_image


SIZE_PRESETS = {
    ("16:9", "2k"): (2560, 1440),
    ("16:9", "4k"): (3840, 2160),
    ("4:3", "2k"): (2048, 1536),
    ("4:3", "4k"): (4096, 3072),
    ("1:1", "2k"): (1440, 1440),
    ("1:1", "4k"): (2160, 2160),
    ("9:16", "2k"): (1440, 2560),
    ("9:16", "4k"): (2160, 3840),
}


class ImageGenerationRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    image: Optional[str] = None
    n: int = Field(default=1, ge=1, le=1)
    size: Optional[str] = None
    aspect_ratio: Optional[str] = None
    resolution: Optional[str] = None
    width: Optional[int] = Field(default=None, ge=64)
    height: Optional[int] = Field(default=None, ge=64)
    num_inference_steps: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = None
    response_format: str = "url"


class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageResponse(BaseModel):
    created: int
    data: list[ImageData]


settings = get_settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FLUX.2 Klein KV OpenAI Image API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")

_service = FluxImageService(settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.model_path}


@app.post("/v1/images/generations/", response_model=ImageResponse)
async def create_image_generation(request: Request, app_settings: Settings = Depends(get_settings)) -> ImageResponse:
    payload = await _parse_generation_request(request)
    reference_image = string_to_image(payload.image)
    return await _run_image_request(payload=payload, reference_image=reference_image, app_settings=app_settings)


@app.post("/v1/images/edits/", response_model=ImageResponse)
async def create_image_edit(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: Optional[UploadFile] = File(default=None),
    model: Optional[str] = Form(default=None),
    n: int = Form(default=1),
    size: Optional[str] = Form(default=None),
    width: Optional[int] = Form(default=None),
    height: Optional[int] = Form(default=None),
    aspect_ratio: Optional[str] = Form(default=None),
    resolution: Optional[str] = Form(default=None),
    num_inference_steps: Optional[int] = Form(default=None),
    seed: Optional[int] = Form(default=None),
    response_format: str = Form(default="url"),
    app_settings: Settings = Depends(get_settings),
) -> ImageResponse:
    if n != 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only n=1 is supported")
    if mask is not None:
        await mask.read()

    payload = ImageGenerationRequest(
        model=model,
        prompt=prompt,
        n=n,
        size=size,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        seed=seed,
        response_format=response_format,
    )
    reference_image = await upload_file_to_image(image)
    return await _run_image_request(payload=payload, reference_image=reference_image, app_settings=app_settings)


async def _parse_generation_request(request: Request) -> ImageGenerationRequest:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data") or content_type.startswith("application/x-www-form-urlencoded"):
        form = await request.form()
        image_file = form.get("image")
        image_value = form.get("image") if isinstance(form.get("image"), str) else None
        if hasattr(image_file, "read"):
            image = await upload_file_to_image(image_file)
            image_value = image_to_base64_png(image) if image is not None else None

        return ImageGenerationRequest(
            model=_optional_str(form.get("model")),
            prompt=str(form.get("prompt") or ""),
            image=image_value,
            n=int(form.get("n") or 1),
            size=_optional_str(form.get("size")),
            aspect_ratio=_optional_str(form.get("aspect_ratio")),
            resolution=_optional_str(form.get("resolution")),
            width=_optional_int(form.get("width")),
            height=_optional_int(form.get("height")),
            num_inference_steps=_optional_int(form.get("num_inference_steps")),
            seed=_optional_int(form.get("seed")),
            response_format=_optional_str(form.get("response_format")) or "url",
        )

    try:
        body: dict[str, Any] = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body must be JSON or form data") from exc
    return ImageGenerationRequest.model_validate(body)


async def _run_image_request(
    *,
    payload: ImageGenerationRequest,
    reference_image,
    app_settings: Settings,
) -> ImageResponse:
    if not payload.prompt.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="prompt is required")
    if payload.n != 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only n=1 is supported")
    if payload.response_format not in {"url", "b64_json"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="response_format must be 'url' or 'b64_json'")

    output_width, output_height = _resolve_dimensions(payload, app_settings)
    generation_width, generation_height = _resolve_generation_dimensions(output_width, output_height, app_settings)
    image = await _service.generate(
        prompt=payload.prompt,
        image=reference_image,
        width=generation_width,
        height=generation_height,
        num_inference_steps=payload.num_inference_steps or app_settings.num_inference_steps,
        seed=payload.seed,
    )
    if image.size != (output_width, output_height):
        image = image.resize((output_width, output_height))

    data = ImageData(revised_prompt=payload.prompt)
    if payload.response_format == "b64_json":
        data.b64_json = image_to_base64_png(image)
    else:
        filename = f"{int(time.time())}-{uuid.uuid4().hex}.png"
        save_png(image, app_settings.output_dir, filename)
        data.url = public_image_url(app_settings.normalized_public_base_url, filename)

    return ImageResponse(created=int(time.time()), data=[data])


def _resolve_dimensions(payload: ImageGenerationRequest, app_settings: Settings) -> tuple[int, int]:
    if payload.aspect_ratio or payload.resolution:
        aspect_ratio = _normalize_aspect_ratio(payload.aspect_ratio or "1:1")
        resolution = (payload.resolution or "2k").lower()
        if (aspect_ratio, resolution) not in SIZE_PRESETS:
            supported = ", ".join(f"{ratio}/{res}" for ratio, res in SIZE_PRESETS)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported aspect_ratio/resolution. Supported: {supported}",
            )
        return SIZE_PRESETS[(aspect_ratio, resolution)]

    width = payload.width
    height = payload.height
    if payload.size and (width is None or height is None):
        try:
            size_width, size_height = payload.size.lower().split("x", 1)
            width = width or int(size_width)
            height = height or int(size_height)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="size must look like '1024x1024'") from exc
    return width or app_settings.default_width, height or app_settings.default_height


def _resolve_generation_dimensions(output_width: int, output_height: int, app_settings: Settings) -> tuple[int, int]:
    output_pixels = output_width * output_height
    if output_pixels <= app_settings.max_generation_pixels:
        return _multiple_of_16(output_width), _multiple_of_16(output_height)

    scale = (app_settings.max_generation_pixels / output_pixels) ** 0.5
    width = max(64, _multiple_of_16(output_width * scale))
    height = max(64, _multiple_of_16(output_height * scale))
    return width, height


def _normalize_aspect_ratio(value: str) -> str:
    normalized = value.strip().lower().replace("：", ":").replace("x", ":")
    aliases = {
        "square": "1:1",
        "portrait": "9:16",
        "vertical": "9:16",
        "landscape": "16:9",
    }
    return aliases.get(normalized, normalized)


def _multiple_of_16(value: float) -> int:
    return max(64, int(round(value / 16)) * 16)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        raise exc
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"message": str(exc), "type": exc.__class__.__name__}},
    )

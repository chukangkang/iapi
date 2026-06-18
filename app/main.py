import asyncio
import time
import uuid
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.image_utils import image_to_base64_png, string_to_image, upload_file_to_image
from app.storage import ImageStorage
from app.tasks import ImageTask, ImageTaskManager


SIZE_PRESETS = {
    ("16:9", "2k"): (2560, 1440),
    ("16:9", "4k"): (3840, 2160),
    ("4:3", "2k"): (2048, 1536),
    ("4:3", "4k"): (4096, 3072),
    ("1:1", "2k"): (1440, 1440),
    ("1:1", "4k"): (4096, 4096),
    ("9:16", "2k"): (1440, 2560),
    ("9:16", "4k"): (2160, 3840),
}

PROMPT_PARAM_PATTERN = re.compile(r"(?P<key>enhance_mode|upscale_fit_mode|aspect_ratio|resolution|size|width|height|qwen_edit_strength)\s*=\s*(?P<value>[^\s,;\]\)]+)", re.IGNORECASE)
SHANGHAI_TIMEZONE = timezone(timedelta(hours=8))
PERSON_PROMPT_PATTERN = re.compile(r"人|男|女|孩|老人|头像|肖像|portrait|person|people|man|woman|girl|boy|child|face", re.IGNORECASE)
ETHNICITY_PROMPT_PATTERN = re.compile(r"中国|华人|亚洲|东亚|欧美|美国|欧洲|白人|黑人|日本|韩国|外国|Chinese|Asian|East Asian|Western|American|European|Caucasian|Black|African|Japanese|Korean|foreigner", re.IGNORECASE)


class ImageGenerationRequest(BaseModel):
    model: Optional[str] = None
    task_id: Optional[str] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
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
    enhance_mode: Optional[str] = None
    upscale_fit_mode: Optional[str] = None
    flux_refine_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    qwen_edit_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageResponse(BaseModel):
    created: str
    data: list[ImageData]


class ImageTaskResponse(BaseModel):
    id: str
    object: str = "image.task"
    status: str
    created: str
    updated: str
    url: str


class ImageTaskResultResponse(BaseModel):
    id: str
    object: str = "image.task"
    status: str
    queue_position: Optional[int] = None
    created: str
    updated: str
    started: Optional[str] = None
    completed: Optional[str] = None
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    result: Optional[ImageResponse] = None
    error: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False


settings = get_settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)

_service: Optional[Any] = None
_qwen_edit_service: Optional[Any] = None
_upscale_service: Optional[Any] = None
_storage: Optional[ImageStorage] = None


def _get_flux_service() -> Any:
    from app.flux_service import FluxImageService

    global _service
    if _service is None:
        _service = FluxImageService(settings)
    return _service


def _get_qwen_edit_service() -> Any:
    from app.qwen_edit_service import QwenImageEditService

    global _qwen_edit_service
    if _qwen_edit_service is None:
        _qwen_edit_service = QwenImageEditService(settings)
    return _qwen_edit_service


def _get_upscale_service() -> Any:
    from app.upscale_service import ImageUpscaleService

    global _upscale_service
    if _upscale_service is None:
        _upscale_service = ImageUpscaleService(settings)
    return _upscale_service


def _get_storage() -> ImageStorage:
    global _storage
    if _storage is None:
        _storage = ImageStorage(settings)
    return _storage


async def _run_task_payload(payload: object, reference_image) -> ImageResponse:
    return await _run_image_request(
        payload=payload,
        reference_image=reference_image,
        app_settings=settings,
    )


_task_manager = ImageTaskManager(settings, _run_task_payload, ImageGenerationRequest.model_validate)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _validate_runtime_settings(settings)
    await _task_manager.start()
    try:
        yield
    finally:
        await _task_manager.stop()


app = FastAPI(title="FLUX.2 Klein KV OpenAI Image API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")


def _validate_runtime_settings(app_settings: Settings) -> None:
    if app_settings.service_role == "api" and app_settings.task_queue_backend == "memory":
        raise RuntimeError("SERVICE_ROLE=api requires TASK_QUEUE_BACKEND=redis; memory queue has no local worker.")
    if app_settings.service_role == "worker":
        raise RuntimeError("SERVICE_ROLE=worker should be started with `python -m app.worker`, not uvicorn app.main:app.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.model_name, "model_path": settings.model_path}


@app.get("/v1/models")
def list_models(app_settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    created = _format_time(int(time.time()))
    return {
        "object": "list",
        "data": [
            {
                "id": app_settings.model_name,
                "object": "model",
                "created": created,
                "owned_by": "iapi",
            }
        ],
    }


@app.post("/v1/chat/completions")
def create_chat_completion(payload: ChatCompletionRequest, app_settings: Settings = Depends(get_settings)):
    created = _format_time(int(time.time()))
    content = f"Backend is healthy. Model name: {app_settings.model_name}"
    if payload.stream:
        return StreamingResponse(
            _chat_completion_stream(app_settings.model_name, created, content),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": app_settings.model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _chat_completion_stream(model_name: str, created: str, content: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    first_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    content_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    import json

    for chunk in (first_chunk, content_chunk, final_chunk):
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/images/generations")
@app.post("/v1/images/generations/")
async def create_image_generation(request: Request, app_settings: Settings = Depends(get_settings)):
    payload, reference_image = await _parse_image_request(request)
    query_task_id = payload.task_id or _task_id_from_prompt(payload.prompt)
    if query_task_id:
        return await _get_image_task_response(query_task_id)
    return await _submit_image_task(payload, reference_image, app_settings)


@app.post("/v1/images/edits", response_model=ImageTaskResponse)
@app.post("/v1/images/edits/", response_model=ImageTaskResponse)
async def create_image_edit(request: Request, app_settings: Settings = Depends(get_settings)) -> ImageTaskResponse:
    payload, reference_image = await _parse_image_request(request)
    if reference_image is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="image is required")
    return await _submit_image_task(payload, reference_image, app_settings)


@app.get("/v1/images/tasks/{task_id}", response_model=ImageTaskResultResponse)
async def get_image_task(task_id: str) -> ImageTaskResultResponse:
    return await _get_image_task_response(task_id)


async def _get_image_task_response(task_id: str) -> ImageTaskResultResponse:
    task = await _task_manager.get(task_id)
    if task is not None:
        return await _task_to_result_response(task)

    task_metadata = _task_manager.store.get(task_id)
    if task_metadata is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return await _task_metadata_to_result_response(task_metadata)


async def _submit_image_task(payload: ImageGenerationRequest, reference_image, app_settings: Settings) -> ImageTaskResponse:
    _validate_image_payload(payload, app_settings)
    if app_settings.service_role == "api" and app_settings.task_queue_backend == "memory":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SERVICE_ROLE=api requires TASK_QUEUE_BACKEND=redis; memory queue has no worker in api-only mode.",
        )
    try:
        task = await _task_manager.submit(payload, reference_image)
    except asyncio.QueueFull as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Image task queue is full") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit image task: {exc}",
        ) from exc
    return ImageTaskResponse(
        id=task.id,
        status=task.status,
        created=_format_time(task.created),
        updated=_format_time(task.updated),
        url=_task_url(task.id, app_settings),
    )


def _task_url(task_id: str, app_settings: Settings) -> str:
    if app_settings.normalized_task_public_base_url:
        return f"{app_settings.normalized_task_public_base_url}/v1/images/tasks/{task_id}"
    return f"/v1/images/tasks/{task_id}"


def _task_id_from_prompt(prompt: Optional[str]) -> Optional[str]:
    if not prompt:
        return None
    value = prompt.strip()
    if value.startswith("task_id:"):
        value = value.split(":", 1)[1].strip()
    if value.startswith("img-"):
        return value
    return None


async def _task_to_result_response(task: ImageTask) -> ImageTaskResultResponse:
    return ImageTaskResultResponse(
        id=task.id,
        status=task.status,
        queue_position=await _task_manager.queue_position(task.id),
        created=_format_time(task.created),
        updated=_format_time(task.updated),
        started=_format_time(task.started),
        completed=_format_time(task.completed),
        worker_id=task.worker_id,
        worker_name=task.worker_name,
        result=task.result,
        error=task.error,
    )


async def _task_metadata_to_result_response(task: dict[str, Any]) -> ImageTaskResultResponse:
    result = ImageResponse.model_validate(task["result"]) if task.get("result") else None
    return ImageTaskResultResponse(
        id=task["id"],
        status=task["status"],
        queue_position=await _task_manager.queue_position(task["id"]),
        created=_format_time(task["created"]),
        updated=_format_time(task["updated"]),
        started=_format_time(task.get("started")),
        completed=_format_time(task.get("completed")),
        worker_id=task.get("worker_id"),
        worker_name=task.get("worker_name"),
        result=result,
        error=task.get("error"),
    )


async def _parse_generation_request(request: Request) -> ImageGenerationRequest:
    payload, _ = await _parse_image_request(request)
    return payload


async def _parse_image_request(request: Request) -> tuple[ImageGenerationRequest, Optional[Any]]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data") or content_type.startswith("application/x-www-form-urlencoded"):
        form = await request.form()
        image_file = form.get("image")
        image_value = form.get("image") if isinstance(form.get("image"), str) else None
        reference_image = None
        if hasattr(image_file, "read"):
            reference_image = await upload_file_to_image(image_file)
            image_value = image_to_base64_png(reference_image) if reference_image is not None else None

        mask = form.get("mask")
        if hasattr(mask, "read"):
            await mask.read()

        payload = ImageGenerationRequest(
            model=_optional_str(form.get("model")),
            task_id=_optional_str(form.get("task_id")),
            prompt=str(form.get("prompt") or ""),
            negative_prompt=_optional_str(form.get("negative_prompt")),
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
            enhance_mode=_optional_str(form.get("enhance_mode")),
            upscale_fit_mode=_optional_str(form.get("upscale_fit_mode")),
            flux_refine_strength=_optional_float(form.get("flux_refine_strength")),
            qwen_edit_strength=_optional_float(form.get("qwen_edit_strength")),
        )
        if reference_image is None:
            reference_image = string_to_image(payload.image)
        _apply_prompt_params(payload)
        return payload, reference_image

    try:
        body: dict[str, Any] = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body must be JSON or form data") from exc
    payload = ImageGenerationRequest.model_validate(body)
    _apply_prompt_params(payload)
    return payload, string_to_image(payload.image)


async def _run_image_request(
    *,
    payload: ImageGenerationRequest,
    reference_image,
    app_settings: Settings,
) -> ImageResponse:
    _validate_image_payload(payload, app_settings)

    output_width, output_height = _resolve_dimensions(payload, app_settings)
    enhance_mode = _resolve_enhance_mode(payload, app_settings)
    upscale_fit_mode = _resolve_upscale_fit_mode(payload, app_settings)
    prompt = _enhance_prompt(payload.prompt, reference_image, app_settings)
    negative_prompt = _merge_negative_prompt(app_settings.default_negative_prompt, payload.negative_prompt)
    metadata = {
        "enhance_mode": enhance_mode,
        "prompt": prompt,
        "original_prompt": payload.prompt,
        "target_width": output_width,
        "target_height": output_height,
        "source_width": reference_image.width if reference_image is not None else None,
        "source_height": reference_image.height if reference_image is not None else None,
        "upscale_fit_mode": upscale_fit_mode,
        "upscale_fill_color": app_settings.upscale_fill_color,
    }
    if negative_prompt:
        metadata["negative_prompt"] = negative_prompt
    if enhance_mode == "pixel":
        metadata["pixel_sharpen_enabled"] = app_settings.pixel_sharpen_enabled
        metadata["pixel_sharpen_percent"] = app_settings.pixel_sharpen_percent
    if enhance_mode in {"realesrgan", "realesrgan_flux", "qwen_edit_realesrgan", "qwen_unblur_upscale_realesrgan"}:
        metadata["realesrgan_max_passes"] = app_settings.realesrgan_max_passes
        metadata["realesrgan_denoise_strength"] = app_settings.realesrgan_denoise_strength
    if reference_image is not None and enhance_mode in {"qwen_edit", "qwen_edit_realesrgan", "qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"}:
        generation_width, generation_height = _resolve_qwen_edit_dimensions(output_width, output_height, app_settings)
        qwen_strength = payload.qwen_edit_strength if payload.qwen_edit_strength is not None else app_settings.qwen_edit_strength
        is_unblur_upscale = enhance_mode in {"qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"}
        qwen_prompt = _qwen_unblur_upscale_prompt(prompt, app_settings) if is_unblur_upscale else prompt
        metadata["qwen_edit_model_path"] = app_settings.qwen_edit_model_path
        metadata["qwen_edit_strength"] = qwen_strength
        metadata["qwen_edit_prompt"] = qwen_prompt
        metadata["qwen_edit_width"] = generation_width
        metadata["qwen_edit_height"] = generation_height
        if is_unblur_upscale:
            metadata["qwen_unblur_upscale_lora_path"] = app_settings.qwen_unblur_upscale_lora_path
            metadata["qwen_unblur_upscale_lora_weight_name"] = app_settings.qwen_unblur_upscale_lora_weight_name
            metadata["qwen_unblur_upscale_lora_scale"] = app_settings.qwen_unblur_upscale_lora_scale
        image = await _get_qwen_edit_service().edit(
            prompt=qwen_prompt,
            negative_prompt=negative_prompt,
            image=reference_image,
            width=generation_width,
            height=generation_height,
            num_inference_steps=payload.num_inference_steps or app_settings.qwen_edit_steps,
            seed=payload.seed,
            guidance_scale=app_settings.qwen_edit_guidance_scale,
            strength=qwen_strength,
            lora_path=app_settings.qwen_unblur_upscale_lora_path if is_unblur_upscale else None,
            lora_weight_name=app_settings.qwen_unblur_upscale_lora_weight_name if is_unblur_upscale else None,
            lora_scale=app_settings.qwen_unblur_upscale_lora_scale,
        )
        upscale_method = "realesrgan" if enhance_mode in {"qwen_edit_realesrgan", "qwen_unblur_upscale_realesrgan"} else "pixel"
        image = await _get_upscale_service().upscale(
            image,
            width=output_width,
            height=output_height,
            method=upscale_method,
            fit_mode=upscale_fit_mode,
        )
        metadata["output_width"] = image.width
        metadata["output_height"] = image.height
        return _image_response(image, payload, metadata)

    if reference_image is not None and enhance_mode in {"pixel", "realesrgan", "realesrgan_flux"}:
        upscale_method = "realesrgan" if enhance_mode in {"realesrgan", "realesrgan_flux"} else "pixel"
        image = await _get_upscale_service().upscale(
            reference_image,
            width=output_width,
            height=output_height,
            method=upscale_method,
            fit_mode=upscale_fit_mode,
        )
        if enhance_mode != "realesrgan_flux":
            metadata["output_width"] = image.width
            metadata["output_height"] = image.height
            return _image_response(image, payload, metadata)

        reference_image = image

    generation_width, generation_height = _resolve_generation_dimensions(output_width, output_height, app_settings)
    image = await _get_flux_service().generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=reference_image,
        width=generation_width,
        height=generation_height,
        num_inference_steps=payload.num_inference_steps or app_settings.num_inference_steps,
        seed=payload.seed,
        strength=(payload.flux_refine_strength or app_settings.flux_refine_strength) if reference_image is not None else None,
    )
    if image.size != (output_width, output_height):
        image = image.resize((output_width, output_height))

    metadata["output_width"] = image.width
    metadata["output_height"] = image.height
    return _image_response(image, payload, metadata)


def _image_response(image, payload: ImageGenerationRequest, metadata: Optional[dict[str, Any]] = None) -> ImageResponse:
    data = ImageData(revised_prompt=payload.prompt, metadata=metadata or {})
    if payload.response_format == "b64_json":
        data.b64_json = image_to_base64_png(image)
    else:
        filename = f"{int(time.time())}-{uuid.uuid4().hex}.png"
        data.url = _get_storage().store_png(image, filename).url

    return ImageResponse(created=_format_time(int(time.time())), data=[data])


def _format_time(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=SHANGHAI_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def _validate_image_payload(payload: ImageGenerationRequest, app_settings: Settings) -> None:
    if not payload.prompt or not payload.prompt.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="prompt is required")
    if payload.n != 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only n=1 is supported")
    if payload.response_format not in {"url", "b64_json"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="response_format must be 'url' or 'b64_json'")
    if payload.enhance_mode and payload.enhance_mode not in {"flux", "pixel", "realesrgan", "realesrgan_flux", "qwen_edit", "qwen_edit_realesrgan", "qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="enhance_mode must be one of: flux, pixel, realesrgan, realesrgan_flux, qwen_edit, qwen_edit_realesrgan, qwen_unblur_upscale, qwen_unblur_upscale_realesrgan",
        )
    if payload.upscale_fit_mode and payload.upscale_fit_mode not in {"stretch", "contain", "cover"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="upscale_fit_mode must be one of: stretch, contain, cover",
        )
    if payload.enhance_mode in {"qwen_edit", "qwen_edit_realesrgan", "qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"} and not payload.image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image is required for Qwen Edit enhance modes.",
        )
    if payload.enhance_mode in {"realesrgan", "realesrgan_flux", "qwen_edit_realesrgan", "qwen_unblur_upscale_realesrgan"} and app_settings.service_role != "api":
        from app.upscale_service import realesrgan_available, realesrgan_import_error

        if not realesrgan_available():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Real-ESRGAN dependencies cannot be imported. Install/fix realesrgan and basicsr, or use enhance_mode=pixel. Import error: {realesrgan_import_error()}",
            )
    if payload.enhance_mode in {"realesrgan", "realesrgan_flux", "qwen_edit_realesrgan", "qwen_unblur_upscale_realesrgan"} and not app_settings.realesrgan_model_path.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="REALESRGAN_MODEL_PATH is required for enhance_mode=realesrgan or realesrgan_flux.",
        )


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


def _resolve_qwen_edit_dimensions(output_width: int, output_height: int, app_settings: Settings) -> tuple[int, int]:
    if app_settings.qwen_edit_scale_to_side == "shortest":
        scale = app_settings.qwen_edit_scale_to_length / min(output_width, output_height)
    else:
        scale = app_settings.qwen_edit_scale_to_length / max(output_width, output_height)
    width = max(64, _round_to_multiple(output_width * scale, app_settings.qwen_edit_round_to_multiple))
    height = max(64, _round_to_multiple(output_height * scale, app_settings.qwen_edit_round_to_multiple))
    return width, height


def _round_to_multiple(value: float, multiple: int) -> int:
    if multiple <= 1:
        return int(round(value))
    return int(round(value / multiple) * multiple)


def _resolve_enhance_mode(payload: ImageGenerationRequest, app_settings: Settings) -> str:
    return payload.enhance_mode or app_settings.default_enhance_mode


def _resolve_upscale_fit_mode(payload: ImageGenerationRequest, app_settings: Settings) -> str:
    return payload.upscale_fit_mode or app_settings.upscale_fit_mode


def _merge_negative_prompt(default_negative_prompt: str, request_negative_prompt: Optional[str]) -> Optional[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for prompt in (default_negative_prompt, request_negative_prompt):
        for term in _split_negative_prompt(prompt):
            normalized = term.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            terms.append(term)
    return ", ".join(terms) if terms else None


def _split_negative_prompt(prompt: Optional[str]) -> list[str]:
    if not prompt:
        return []
    return [term.strip() for term in re.split(r"[,，;；\n]+", prompt) if term.strip()]


def _enhance_prompt(prompt: Optional[str], reference_image, app_settings: Settings) -> str:
    user_prompt = (prompt or "").strip()
    if not app_settings.prompt_enhance_enabled or reference_image is not None:
        return user_prompt
    if len(user_prompt) > app_settings.prompt_enhance_short_max_chars:
        return user_prompt

    additions = []
    if PERSON_PROMPT_PATTERN.search(user_prompt) and not ETHNICITY_PROMPT_PATTERN.search(user_prompt):
        additions.append(app_settings.prompt_enhance_person_suffix)
    if app_settings.prompt_enhance_suffix:
        additions.append(app_settings.prompt_enhance_suffix)
    if not additions:
        return user_prompt
    return ", ".join([user_prompt, *additions])


def _qwen_unblur_upscale_prompt(prompt: Optional[str], app_settings: Settings) -> str:
    trigger_prompt = app_settings.qwen_unblur_upscale_trigger_prompt.strip()
    user_prompt = (prompt or "").strip()
    if not user_prompt:
        return trigger_prompt
    if trigger_prompt and trigger_prompt.lower() not in user_prompt.lower():
        return f"{trigger_prompt}, {user_prompt}"
    return user_prompt


def _apply_prompt_params(payload: ImageGenerationRequest) -> None:
    if not payload.prompt:
        return
    for match in PROMPT_PARAM_PATTERN.finditer(payload.prompt):
        key = match.group("key").lower()
        value = match.group("value").strip().strip('"\'')
        if key == "enhance_mode" and not payload.enhance_mode:
            payload.enhance_mode = value.lower()
        elif key == "upscale_fit_mode" and not payload.upscale_fit_mode:
            payload.upscale_fit_mode = value.lower()
        elif key == "qwen_edit_strength" and payload.qwen_edit_strength is None:
            payload.qwen_edit_strength = float(value)
        elif key == "aspect_ratio" and not payload.aspect_ratio:
            payload.aspect_ratio = value
        elif key == "resolution" and not payload.resolution:
            payload.resolution = value.lower()
        elif key == "size" and not payload.size:
            payload.size = value
        elif key == "width" and payload.width is None:
            payload.width = int(value)
        elif key == "height" and payload.height is None:
            payload.height = int(value)


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


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        raise exc
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"message": str(exc), "type": exc.__class__.__name__}},
    )

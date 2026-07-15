import asyncio
import logging
import math
import time
import uuid
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_serializer

from app.config import Settings, get_settings
from app.image_utils import image_to_base64_png, string_list_to_images, string_to_image, upload_file_to_image
from app.storage import ImageStorage
from app.tasks import ImageTask, ImageTaskManager


logger = logging.getLogger(__name__)


SIZE_PRESETS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

EDIT_SIZE_PRESETS = {
    ("16:9", "2k"): (2560, 1440),
    ("16:9", "4k"): (3840, 2160),
    ("4:3", "2k"): (2048, 1536),
    ("4:3", "4k"): (4096, 3072),
    ("1:1", "2k"): (1440, 1440),
    ("1:1", "4k"): (4096, 4096),
    ("9:16", "2k"): (1440, 2560),
    ("9:16", "4k"): (2160, 3840),
}

PROMPT_PARAM_PATTERN = re.compile(r"(?P<key>enhance_mode|upscale_fit_mode|aspect_ratio|resolution|size|width|height|qwen_edit_strength|face_enhance)\s*=\s*(?P<value>[^\s,;\]\)]+)", re.IGNORECASE)
SHANGHAI_TIMEZONE = timezone(timedelta(hours=8))


class ImageGenerationRequest(BaseModel):
    endpoint: str = "generations"
    model: Optional[str] = None
    task_id: Optional[str] = None
    prompt: Optional[str] = None
    original_prompt: Optional[str] = None
    prompt_enhance: Optional[bool] = None
    negative_prompt: Optional[str] = None
    image: Optional[str | list[str]] = None
    n: int = Field(default=1, ge=1, le=1)
    size: Optional[str] = None
    aspect_ratio: Optional[str] = None
    resolution: Optional[str] = None
    width: Optional[int] = Field(default=None, ge=64)
    height: Optional[int] = Field(default=None, ge=64)
    num_inference_steps: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = 42
    response_format: str = "url"
    enhance_mode: Optional[str] = None
    upscale_fit_mode: Optional[str] = None
    face_enhance: Optional[bool] = None
    qwen_edit_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("response_format", mode="before")
    @classmethod
    def _normalize_response_format(cls, value: Any) -> str:
        if value is None:
            return "url"
        normalized = str(value).strip().lower()
        if normalized in {"base64", "b64"}:
            return "b64_json"
        return normalized or "url"


class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if data.get("metadata") is None:
            data.pop("metadata", None)
        return data


class ImageInputTokensDetails(BaseModel):
    text_tokens: int = 0
    image_tokens: int = 0


class ImageUsage(BaseModel):
    total_tokens: int = 10000
    input_tokens: int = 5000
    output_tokens: int = 5000
    input_tokens_details: ImageInputTokensDetails = Field(default_factory=ImageInputTokensDetails)


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
    usage: ImageUsage = Field(default_factory=ImageUsage)


class ImageTaskResultResponse(BaseModel):
    id: str
    object: str = "image.task"
    status: str
    queue_position: Optional[int] = None
    created: str
    updated: str
    started: Optional[str] = None
    completed: Optional[str] = None
    duration: Optional[int] = None
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
_qwen_image_service: Optional[Any] = None
_qwen_edit_service: Optional[Any] = None
_upscale_service: Optional[Any] = None
_storage: Optional[ImageStorage] = None
_prompt_enhancer: Optional[Any] = None


def _get_qwen_image_service() -> Any:
    from app.qwen_image_service import QwenImageService

    global _qwen_image_service, _qwen_edit_service
    if _qwen_edit_service is not None:
        _qwen_edit_service.unload()
        _qwen_edit_service = None
    if _qwen_image_service is None:
        _qwen_image_service = QwenImageService(settings)
    return _qwen_image_service


def _get_qwen_edit_service() -> Any:
    from app.qwen_edit_service import QwenImageEditService

    global _qwen_edit_service, _qwen_image_service
    if _qwen_image_service is not None:
        _qwen_image_service.unload()
        _qwen_image_service = None
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


def _get_prompt_enhancer(app_settings: Settings) -> Any:
    from app.prompt_enhancer import PromptEnhancer

    global _prompt_enhancer
    if _prompt_enhancer is None or _prompt_enhancer.settings is not app_settings:
        _prompt_enhancer = PromptEnhancer(app_settings)
    return _prompt_enhancer


async def _run_task_payload(payload: object, reference_image) -> ImageResponse:
    return await _run_image_request(
        payload=payload,
        reference_image=reference_image,
        app_settings=settings,
    )


async def _prepare_task_payload(payload: object, reference_image) -> None:
    await _prepare_image_request(
        payload=payload,
        reference_image=reference_image,
        app_settings=settings,
    )


def _task_affinity_key(payload: object, has_reference_image: bool) -> str:
    return _image_request_affinity_key(
        payload=payload,
        has_reference_image=has_reference_image,
        app_settings=settings,
    )


_task_manager = ImageTaskManager(settings, _run_task_payload, ImageGenerationRequest.model_validate, _prepare_task_payload, _task_affinity_key)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _validate_runtime_settings(settings)
    await _task_manager.start()
    try:
        yield
    finally:
        await _task_manager.stop()


app = FastAPI(title="Qwen Image OpenAI API", version="1.0.0", lifespan=lifespan)
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
    _get_prompt_enhancer(app_settings).validate_configuration()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.model_name, "model_path": settings.model_path}


@app.get("/v1/models")
def list_models(app_settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    created = _format_time(int(time.time()))
    model_ids = [app_settings.model_name]
    if app_settings.qwen_image_model_name not in model_ids:
        model_ids.append(app_settings.qwen_image_model_name)
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "iapi",
            }
            for model_id in model_ids
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
    payload.endpoint = "generations"
    query_task_id = payload.task_id or _task_id_from_prompt(payload.prompt)
    if query_task_id:
        return await _get_image_task_response(query_task_id)
    return await _submit_image_task(payload, reference_image, app_settings)


@app.post("/v1/images/edits", response_model=ImageTaskResponse)
@app.post("/v1/images/edits/", response_model=ImageTaskResponse)
async def create_image_edit(request: Request, app_settings: Settings = Depends(get_settings)) -> ImageTaskResponse:
    payload, reference_image = await _parse_image_request(request)
    payload.endpoint = "edits"
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
    await _enhance_image_prompt(payload, app_settings)
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
        duration=_task_duration(task.started, task.completed, task.status),
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
        duration=_task_duration(task.get("started"), task.get("completed"), task["status"]),
        worker_id=task.get("worker_id"),
        worker_name=task.get("worker_name"),
        result=result,
        error=task.get("error"),
    )


def _task_duration(started: Optional[int], completed: Optional[int], status: str) -> Optional[int]:
    if started is None:
        return None
    end_time = completed or (int(time.time()) if status == "running" else None)
    if end_time is None:
        return None
    return max(0, end_time - started)


async def _parse_generation_request(request: Request) -> ImageGenerationRequest:
    payload, _ = await _parse_image_request(request)
    return payload


async def _parse_image_request(request: Request) -> tuple[ImageGenerationRequest, Optional[Any]]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data") or content_type.startswith("application/x-www-form-urlencoded"):
        form = await request.form()
        image_items = form.getlist("image")
        image_files = [item for item in image_items if hasattr(item, "read")]
        image_values = [str(item) for item in image_items if isinstance(item, str) and str(item).strip()]
        image_file = image_files[0] if image_files else form.get("image")
        image_value: Optional[str | list[str]] = image_values[0] if len(image_values) == 1 else image_values or None
        reference_image = None
        if image_files:
            reference_images = []
            for image_file in image_files:
                uploaded_image = await upload_file_to_image(image_file)
                if uploaded_image is not None:
                    reference_images.append(uploaded_image)
            if len(reference_images) == 1:
                reference_image = reference_images[0]
                image_value = image_to_base64_png(reference_image)
            elif reference_images:
                reference_image = reference_images
                image_value = [image_to_base64_png(image) for image in reference_images]

        mask = form.get("mask")
        if hasattr(mask, "read"):
            await mask.read()

        payload = ImageGenerationRequest(
            model=_optional_str(form.get("model")),
            task_id=_optional_str(form.get("task_id")),
            prompt=str(form.get("prompt") or ""),
            prompt_enhance=_optional_bool(form.get("prompt_enhance")),
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
            face_enhance=_optional_bool(form.get("face_enhance")),
            qwen_edit_strength=_optional_float(form.get("qwen_edit_strength")),
        )
        if reference_image is None:
            reference_image = _payload_image_to_reference(payload.image)
        _apply_prompt_params(payload)
        return payload, reference_image

    try:
        body: dict[str, Any] = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body must be JSON or form data") from exc
    payload = ImageGenerationRequest.model_validate(body)
    _apply_prompt_params(payload)
    return payload, _payload_image_to_reference(payload.image)


async def _run_image_request(
    *,
    payload: ImageGenerationRequest,
    reference_image,
    app_settings: Settings,
) -> ImageResponse:
    _validate_image_payload(payload, app_settings)

    primary_reference_image = _primary_reference_image(reference_image)
    reference_image_count = _reference_image_count(reference_image)
    output_width, output_height = _resolve_dimensions(payload, app_settings, primary_reference_image)
    enhance_mode = _resolve_enhance_mode(payload, app_settings)
    upscale_fit_mode = _resolve_upscale_fit_mode(payload, app_settings)
    face_enhance = _resolve_face_enhance(payload, app_settings)
    prompt = (payload.prompt or "").strip()
    negative_prompt = _resolve_negative_prompt(app_settings)
    metadata = {
        "enhance_mode": enhance_mode,
        "prompt": prompt,
        "original_prompt": payload.original_prompt or payload.prompt,
        "prompt_enhanced": payload.original_prompt is not None,
        "target_width": output_width,
        "target_height": output_height,
        "source_width": primary_reference_image.width if primary_reference_image is not None else None,
        "source_height": primary_reference_image.height if primary_reference_image is not None else None,
        "source_image_count": reference_image_count,
        "upscale_fit_mode": upscale_fit_mode,
        "upscale_fill_color": app_settings.upscale_fill_color,
        "face_enhance": face_enhance,
    }
    if negative_prompt:
        metadata["negative_prompt"] = negative_prompt
    if enhance_mode == "pixel":
        metadata["pixel_sharpen_enabled"] = app_settings.pixel_sharpen_enabled
        metadata["pixel_sharpen_percent"] = app_settings.pixel_sharpen_percent
    if enhance_mode in {"realesrgan", "qwen_edit_realesrgan", "qwen_unblur_upscale_realesrgan"}:
        metadata["realesrgan_max_passes"] = app_settings.realesrgan_max_passes
        metadata["realesrgan_denoise_strength"] = app_settings.realesrgan_denoise_strength
    if reference_image is not None and enhance_mode in {"qwen_edit", "qwen_edit_realesrgan", "qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"}:
        if reference_image_count > 1 and enhance_mode != "qwen_edit":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Multiple input images are only supported for qwen_edit mode.")
        generation_width, generation_height = _resolve_qwen_edit_dimensions(output_width, output_height, app_settings)
        qwen_strength = payload.qwen_edit_strength if payload.qwen_edit_strength is not None else app_settings.qwen_edit_strength
        is_unblur_upscale = enhance_mode in {"qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"}
        qwen_prompt = _resolve_qwen_edit_prompt(
            prompt=prompt,
            original_prompt=payload.original_prompt,
            is_unblur_upscale=is_unblur_upscale,
            trigger_prompt=app_settings.qwen_unblur_upscale_trigger_prompt,
        )
        logger.info(
            "Final Qwen Edit prompt used by worker: restored_original=%s prompt=%s",
            payload.original_prompt is not None,
            qwen_prompt,
        )
        metadata["qwen_edit_model_path"] = app_settings.qwen_edit_model_path
        if not payload.enhance_mode and _is_qwen_image_model(payload.model, app_settings):
            metadata["qwen_image_i2i_fallback"] = "qwen_edit"
        metadata["qwen_edit_strength"] = qwen_strength
        metadata["qwen_edit_prompt"] = qwen_prompt
        metadata["qwen_edit_task_type"] = "image_merge" if reference_image_count > 1 else "image_to_image"
        metadata["qwen_edit_width"] = generation_width
        metadata["qwen_edit_height"] = generation_height
        if is_unblur_upscale:
            metadata["qwen_unblur_upscale_lora_path"] = app_settings.qwen_unblur_upscale_lora_path
            metadata["qwen_unblur_upscale_lora_weight_name"] = app_settings.qwen_unblur_upscale_lora_weight_name
            metadata["qwen_unblur_upscale_lora_scale"] = app_settings.qwen_unblur_upscale_lora_scale
        unblur_lora_path, unblur_lora_weight_name = _resolve_qwen_unblur_lora(is_unblur_upscale, app_settings)
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
            lora_path=unblur_lora_path,
            lora_weight_name=unblur_lora_weight_name,
            lora_scale=app_settings.qwen_unblur_upscale_lora_scale,
        )
        if is_unblur_upscale and primary_reference_image is not None:
            qwen_edit_service = _get_qwen_edit_service()
            image = qwen_edit_service.align_to_reference(image, primary_reference_image)
            metadata["qwen_unblur_upscale_alignment_enabled"] = app_settings.qwen_unblur_upscale_alignment_enabled
        upscale_method = "realesrgan" if enhance_mode in {"qwen_edit_realesrgan", "qwen_unblur_upscale_realesrgan"} else "pixel"
        image = await _get_upscale_service().upscale(
            image,
            width=output_width,
            height=output_height,
            method=upscale_method,
            fit_mode=upscale_fit_mode,
            face_enhance=face_enhance,
        )
        metadata["output_width"] = image.width
        metadata["output_height"] = image.height
        return _image_response(image, payload, metadata)

    if primary_reference_image is not None and enhance_mode in {"pixel", "realesrgan"}:
        upscale_method = "realesrgan" if enhance_mode == "realesrgan" else "pixel"
        image = await _get_upscale_service().upscale(
            primary_reference_image,
            width=output_width,
            height=output_height,
            method=upscale_method,
            fit_mode=upscale_fit_mode,
            face_enhance=face_enhance,
        )
        metadata["output_width"] = image.width
        metadata["output_height"] = image.height
        return _image_response(image, payload, metadata)

    generation_width, generation_height = _resolve_generation_dimensions(output_width, output_height, app_settings)
    metadata["qwen_image_model_path"] = app_settings.qwen_image_model_path
    qwen_image_service = _get_qwen_image_service()
    metadata["qwen_image_steps"] = payload.num_inference_steps or app_settings.qwen_image_steps
    metadata["qwen_image_true_cfg_scale"] = app_settings.qwen_image_true_cfg_scale
    metadata["qwen_image_task_type"] = "image_to_image" if primary_reference_image is not None else "text_to_image"
    metadata["qwen_image_width"] = generation_width
    metadata["qwen_image_height"] = generation_height
    image = await qwen_image_service.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=primary_reference_image,
        width=generation_width,
        height=generation_height,
        num_inference_steps=payload.num_inference_steps or app_settings.qwen_image_steps,
        seed=payload.seed,
    )
    if image.size != (output_width, output_height):
        image = image.resize((output_width, output_height))

    metadata["output_width"] = image.width
    metadata["output_height"] = image.height
    return _image_response(image, payload, metadata)


async def _prepare_image_request(
    *,
    payload: ImageGenerationRequest,
    reference_image,
    app_settings: Settings,
) -> None:
    _validate_image_payload(payload, app_settings)
    primary_reference_image = _primary_reference_image(reference_image)
    reference_image_count = _reference_image_count(reference_image)
    output_width, output_height = _resolve_dimensions(payload, app_settings, primary_reference_image)
    enhance_mode = _resolve_enhance_mode(payload, app_settings)

    if reference_image is not None and enhance_mode in {"qwen_edit", "qwen_edit_realesrgan", "qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"}:
        if reference_image_count > 1 and enhance_mode != "qwen_edit":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Multiple input images are only supported for qwen_edit mode.")
        is_unblur_upscale = enhance_mode in {"qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"}
        unblur_lora_path, unblur_lora_weight_name = _resolve_qwen_unblur_lora(is_unblur_upscale, app_settings)
        await _get_qwen_edit_service().prepare(
            lora_path=unblur_lora_path,
            lora_weight_name=unblur_lora_weight_name,
        )
        if enhance_mode in {"qwen_edit_realesrgan", "qwen_unblur_upscale_realesrgan"}:
            await _get_upscale_service().prepare(method="realesrgan")
        return

    if primary_reference_image is not None and enhance_mode in {"pixel", "realesrgan"}:
        if enhance_mode == "realesrgan":
            await _get_upscale_service().prepare(method="realesrgan")
        return

    await _get_qwen_image_service().prepare()


def _image_request_affinity_key(
    *,
    payload: ImageGenerationRequest,
    has_reference_image: bool,
    app_settings: Settings,
) -> str:
    enhance_mode = _resolve_enhance_mode(payload, app_settings)
    if has_reference_image and enhance_mode in {"qwen_edit", "qwen_edit_realesrgan", "qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"}:
        lora_key = "unblur" if enhance_mode in {"qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"} else "base"
        upscale_key = "realesrgan" if enhance_mode in {"qwen_edit_realesrgan", "qwen_unblur_upscale_realesrgan"} else "pixel"
        return f"qwen_edit:{app_settings.qwen_edit_model_path}:{lora_key}:{upscale_key}"
    if enhance_mode == "qwen_image":
        return f"qwen_image:{app_settings.qwen_image_model_path}"
    if has_reference_image and enhance_mode in {"pixel", "realesrgan"}:
        if enhance_mode == "realesrgan":
            return f"realesrgan:{app_settings.realesrgan_model_path}:{app_settings.realesrgan_model_name}"
        return "pixel"
    return f"qwen_image:{app_settings.qwen_image_model_path}"


def _image_response(image, payload: ImageGenerationRequest, metadata: Optional[dict[str, Any]] = None) -> ImageResponse:
    data = ImageData(revised_prompt=payload.prompt)
    if settings.response_metadata_enabled:
        data.metadata = metadata or {}
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="response_format must be 'url', 'b64_json', or 'base64'")
    if payload.enhance_mode and payload.enhance_mode not in {"qwen_image", "pixel", "realesrgan", "qwen_edit", "qwen_edit_realesrgan", "qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="enhance_mode must be one of: qwen_image, pixel, realesrgan, qwen_edit, qwen_edit_realesrgan, qwen_unblur_upscale, qwen_unblur_upscale_realesrgan",
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
    if isinstance(payload.image, list):
        if not 1 <= len(payload.image) <= 2:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="image must contain one or two images")
        if len(payload.image) == 2 and _resolve_enhance_mode(payload, app_settings) != "qwen_edit":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Two input images are only supported for qwen_edit mode or qwen-image-2512 model.")
    realesrgan_enhance_modes = {"realesrgan", "qwen_edit_realesrgan", "qwen_unblur_upscale_realesrgan"}
    if payload.enhance_mode in realesrgan_enhance_modes and app_settings.service_role != "api":
        from app.upscale_service import realesrgan_available, realesrgan_import_error

        if not realesrgan_available():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Real-ESRGAN dependencies cannot be imported. Install/fix realesrgan and basicsr, or use enhance_mode=pixel. Import error: {realesrgan_import_error()}",
            )

def _resolve_dimensions(payload: ImageGenerationRequest, app_settings: Settings, reference_image=None) -> tuple[int, int]:
    if payload.endpoint == "edits":
        return _resolve_edit_dimensions(payload, app_settings, reference_image)

    if (
        reference_image is not None
        and payload.resolution
        and payload.enhance_mode
        in {"qwen_edit", "qwen_edit_realesrgan", "qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"}
        and not payload.aspect_ratio
    ):
        return _edit_dimensions_from_reference_resolution(reference_image, payload.resolution.lower())

    if (
        reference_image is not None
        and payload.enhance_mode
        in {"qwen_edit", "qwen_edit_realesrgan", "qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"}
        and not payload.aspect_ratio
    ):
        return _dimensions_from_reference_aspect_ratio(
            reference_image,
            (app_settings.default_width, app_settings.default_height),
        )

    if reference_image is not None and not payload.aspect_ratio:
        return SIZE_PRESETS[_closest_preset_aspect_ratio(reference_image, SIZE_PRESETS.keys())]

    aspect_ratio = _normalize_aspect_ratio(payload.aspect_ratio or "1:1")
    if aspect_ratio not in SIZE_PRESETS:
        supported = ", ".join(SIZE_PRESETS)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported aspect_ratio. Supported: {supported}",
        )
    return SIZE_PRESETS[aspect_ratio]


def _resolve_edit_dimensions(payload: ImageGenerationRequest, app_settings: Settings, reference_image=None) -> tuple[int, int]:
    if payload.aspect_ratio or payload.resolution:
        resolution = (payload.resolution or "2k").lower()
        if payload.resolution and not payload.aspect_ratio and reference_image is not None:
            return _edit_dimensions_from_reference_resolution(reference_image, resolution)
        aspect_ratio = _normalize_aspect_ratio(payload.aspect_ratio) if payload.aspect_ratio else _closest_preset_aspect_ratio(reference_image, EDIT_SIZE_PRESETS.keys())
        if (aspect_ratio, resolution) not in EDIT_SIZE_PRESETS:
            supported = ", ".join(f"{ratio}/{res}" for ratio, res in EDIT_SIZE_PRESETS)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported aspect_ratio/resolution for edits. Supported: {supported}",
            )
        return EDIT_SIZE_PRESETS[(aspect_ratio, resolution)]

    width = payload.width
    height = payload.height
    if payload.size and (width is None or height is None):
        try:
            size_width, size_height = payload.size.lower().split("x", 1)
            width = width or int(size_width)
            height = height or int(size_height)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="size must look like '1024x1024'") from exc
    if reference_image is not None and width is None and height is None and not payload.size:
        return _dimensions_from_reference_aspect_ratio(reference_image, (app_settings.default_width, app_settings.default_height))
    return width or app_settings.default_width, height or app_settings.default_height


def _edit_dimensions_from_reference_resolution(reference_image, resolution: str) -> tuple[int, int]:
    long_side_by_resolution = {
        "2k": 2560,
        "4k": 4096,
    }
    long_side = long_side_by_resolution.get(resolution)
    if long_side is None:
        supported = ", ".join(sorted(long_side_by_resolution))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported resolution for edits with reference aspect ratio. Supported: {supported}",
        )
    if reference_image is None or reference_image.width <= 0 or reference_image.height <= 0:
        return long_side, long_side
    if reference_image.width >= reference_image.height:
        width = long_side
        height = _multiple_of_16(long_side * reference_image.height / reference_image.width)
    else:
        height = long_side
        width = _multiple_of_16(long_side * reference_image.width / reference_image.height)
    return width, height


def _dimensions_from_reference_aspect_ratio(reference_image, base_size: tuple[int, int]) -> tuple[int, int]:
    base_width, base_height = base_size
    if reference_image.width <= 0 or reference_image.height <= 0:
        return base_width, base_height
    target_pixels = base_width * base_height
    divisor = math.gcd(reference_image.width, reference_image.height)
    ratio_width = reference_image.width // divisor
    ratio_height = reference_image.height // divisor
    multiplier_quantum = math.lcm(
        16 // math.gcd(ratio_width, 16),
        16 // math.gcd(ratio_height, 16),
    )
    ideal_multiplier = (target_pixels / (ratio_width * ratio_height)) ** 0.5
    multiplier = max(multiplier_quantum, round(ideal_multiplier / multiplier_quantum) * multiplier_quantum)
    width = ratio_width * multiplier
    height = ratio_height * multiplier
    if width > base_width * 4 or height > base_height * 4:
        aspect_ratio = reference_image.width / reference_image.height
        width = _multiple_of_16((target_pixels * aspect_ratio) ** 0.5)
        height = _multiple_of_16(width / aspect_ratio)
    return width, height


def _closest_preset_aspect_ratio(reference_image, preset_keys) -> str:
    if reference_image is None or reference_image.width <= 0 or reference_image.height <= 0:
        return "1:1"
    source_ratio = reference_image.width / reference_image.height
    ratios = sorted(_preset_aspect_ratios(preset_keys))
    return min(ratios, key=lambda ratio: abs(_aspect_ratio_value(ratio) - source_ratio))


def _preset_aspect_ratios(preset_keys) -> set[str]:
    ratios = set()
    for key in preset_keys:
        ratios.add(key[0] if isinstance(key, tuple) else key)
    return ratios


def _aspect_ratio_value(value: str) -> float:
    width, height = _normalize_aspect_ratio(value).split(":", 1)
    return int(width) / int(height)


def _resolve_generation_dimensions(output_width: int, output_height: int, app_settings: Settings) -> tuple[int, int]:
    return output_width, output_height


def _resolve_qwen_edit_dimensions(output_width: int, output_height: int, app_settings: Settings) -> tuple[int, int]:
    if app_settings.qwen_edit_scale_to_length == 0:
        scale = 1.0
    elif app_settings.qwen_edit_scale_to_side == "shortest":
        scale = app_settings.qwen_edit_scale_to_length / min(output_width, output_height)
    else:
        scale = app_settings.qwen_edit_scale_to_length / max(output_width, output_height)
    width = max(64, _round_to_multiple(output_width * scale, app_settings.qwen_edit_round_to_multiple))
    height = max(64, _round_to_multiple(output_height * scale, app_settings.qwen_edit_round_to_multiple))
    return width, height


def _resolve_qwen_unblur_lora(is_unblur_upscale: bool, app_settings: Settings) -> tuple[Optional[str], Optional[str]]:
    if not is_unblur_upscale or not app_settings.qwen_unblur_upscale_lora_enabled:
        return None, None
    return app_settings.qwen_unblur_upscale_lora_path, app_settings.qwen_unblur_upscale_lora_weight_name


def _resolve_qwen_edit_prompt(
    *,
    prompt: str,
    original_prompt: Optional[str],
    is_unblur_upscale: bool,
    trigger_prompt: str,
) -> str:
    original = (original_prompt or "").strip()
    if original:
        return original
    trigger = trigger_prompt.strip()
    if is_unblur_upscale and trigger:
        return trigger
    return prompt.strip()


def _round_to_multiple(value: float, multiple: int) -> int:
    if multiple <= 1:
        return int(round(value))
    return int(round(value / multiple) * multiple)


def _resolve_enhance_mode(payload: ImageGenerationRequest, app_settings: Settings) -> str:
    if payload.enhance_mode:
        return payload.enhance_mode
    if _is_qwen_image_model(payload.model, app_settings):
        if payload.image:
            return "qwen_edit"
        return "qwen_image"
    return app_settings.default_enhance_mode


def _payload_image_to_reference(image: Optional[str | list[str]]) -> Optional[Any]:
    if isinstance(image, list):
        images = string_list_to_images(image)
        if len(images) == 1:
            return images[0]
        return images or None
    return string_to_image(image)


def _primary_reference_image(reference_image) -> Optional[Any]:
    if isinstance(reference_image, list):
        return reference_image[0] if reference_image else None
    return reference_image


def _reference_image_count(reference_image) -> int:
    if isinstance(reference_image, list):
        return len(reference_image)
    return 1 if reference_image is not None else 0


def _is_qwen_image_model(model: Optional[str], app_settings: Settings) -> bool:
    if not model:
        return False
    requested_model = model.strip().casefold()
    return requested_model in {
        app_settings.qwen_image_model_name.casefold(),
        Path(app_settings.qwen_image_model_path).name.casefold(),
        "qwen-image-2512",
        "qwen_image_2512",
    }


def _resolve_upscale_fit_mode(payload: ImageGenerationRequest, app_settings: Settings) -> str:
    return payload.upscale_fit_mode or app_settings.upscale_fit_mode


def _resolve_face_enhance(payload: ImageGenerationRequest, app_settings: Settings) -> bool:
    return payload.face_enhance if payload.face_enhance is not None else app_settings.realesrgan_face_enhance


def _resolve_negative_prompt(app_settings: Settings) -> Optional[str]:
    return app_settings.qwen_image_negative_prompt.strip() or None


async def _enhance_image_prompt(payload: ImageGenerationRequest, app_settings: Settings) -> None:
    should_enhance = payload.prompt_enhance if payload.prompt_enhance is not None else app_settings.prompt_enhancer_enabled
    if not should_enhance or not payload.prompt:
        return
    if payload.enhance_mode in {
        "pixel",
        "realesrgan",
        "qwen_edit",
        "qwen_edit_realesrgan",
        "qwen_unblur_upscale",
        "qwen_unblur_upscale_realesrgan",
    }:
        logger.info("Skipping prompt enhancement for image processing mode: %s", payload.enhance_mode)
        return
    original_prompt = payload.prompt.strip()
    expanded_prompt = await _get_prompt_enhancer(app_settings).enhance(
        original_prompt,
        aspect_ratio=payload.aspect_ratio,
    )
    if expanded_prompt != original_prompt:
        payload.original_prompt = original_prompt
        payload.prompt = expanded_prompt
        logger.info("Final image prompt after enhancement: %s", expanded_prompt)
    else:
        logger.info("Final image prompt unchanged: %s", original_prompt)


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
        elif key == "face_enhance" and payload.face_enhance is None:
            payload.face_enhance = _parse_bool(value)
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


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    return _parse_bool(str(value))


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        raise exc
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"message": str(exc), "type": exc.__class__.__name__}},
    )

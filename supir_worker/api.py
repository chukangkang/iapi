import asyncio
import base64
import binascii
import hmac
import io
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from supir_worker.backend import SupirBackend
from supir_worker.settings import SupirWorkerSettings


class RestoreRequest(BaseModel):
    image: str = Field(min_length=1)
    prompt: str = ""
    width: int = Field(ge=64, le=8192)
    height: int = Field(ge=64, le=8192)


def create_app(settings: SupirWorkerSettings, backend: Any | None = None) -> FastAPI:
    backend = backend or SupirBackend(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if settings.preload and hasattr(backend, "load"):
            await asyncio.to_thread(backend.load)
        yield

    app = FastAPI(title="SUPIR Worker", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ready": bool(backend.ready), "model": backend.model_sign}

    @app.post("/v1/restore")
    async def restore(payload: RestoreRequest, authorization: str | None = Header(default=None)):
        _authorize(settings, authorization)
        if payload.width * payload.height > settings.max_pixels:
            raise HTTPException(status_code=413, detail="Requested output exceeds SUPIR_WORKER_MAX_PIXELS")
        image = _decode_image(payload.image)
        started = time.perf_counter()
        try:
            restored = await asyncio.to_thread(
                backend.restore,
                image,
                prompt=payload.prompt,
                width=payload.width,
                height=payload.height,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        buffer = io.BytesIO()
        restored.convert("RGB").save(buffer, format="PNG")
        return {
            "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "model": backend.model_sign,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "width": restored.width,
            "height": restored.height,
        }

    return app


def _authorize(settings: SupirWorkerSettings, authorization: str | None) -> None:
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _decode_image(encoded: str) -> Image.Image:
    try:
        raw = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="image must be a valid base64 encoded image") from exc


settings = SupirWorkerSettings()
app = create_app(settings)
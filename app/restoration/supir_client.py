import asyncio
import base64
import io
import json
import logging
from urllib.request import Request, urlopen

from PIL import Image

from app.config import Settings


logger = logging.getLogger(__name__)


class SupirClient:
    """Optional client for an isolated SUPIR restoration service."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def available(self) -> bool:
        return self.settings.supir_enabled and bool(self.settings.supir_base_url.strip())

    async def restore(self, image: Image.Image, *, prompt: str, width: int, height: int) -> Image.Image:
        if not self.available:
            return image
        try:
            return await asyncio.to_thread(self._restore_sync, image, prompt=prompt, width=width, height=height)
        except Exception as exc:
            logger.warning("SUPIR service failed; falling back to local restoration: %s", exc)
            return image

    def _restore_sync(self, image: Image.Image, *, prompt: str, width: int, height: int) -> Image.Image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        payload = {
            "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "prompt": prompt,
            "width": width,
            "height": height,
        }
        url = f"{self.settings.supir_base_url.rstrip('/')}/{self.settings.supir_endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.settings.supir_api_key:
            headers["Authorization"] = f"Bearer {self.settings.supir_api_key}"
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.settings.supir_timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        encoded = result.get("image") or result.get("b64_json")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("SUPIR response is missing image/b64_json")
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFilter

from app.config import Settings


class ImageUpscaleService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._upsampler = None

    async def upscale(
        self,
        image: Image.Image,
        *,
        width: int,
        height: int,
        method: str,
    ) -> Image.Image:
        return await asyncio.to_thread(self._upscale_sync, image, width=width, height=height, method=method)

    def _upscale_sync(self, image: Image.Image, *, width: int, height: int, method: str) -> Image.Image:
        image = image.convert("RGB")
        if method == "realesrgan":
            upscaled = self._realesrgan_upscale(image, width=width, height=height)
            if upscaled is not None:
                return upscaled
        upscaled = image.resize((width, height), Image.Resampling.LANCZOS)
        return self._sharpen_pixel_upscale(upscaled) if method == "pixel" else upscaled

    def _sharpen_pixel_upscale(self, image: Image.Image) -> Image.Image:
        if not self.settings.pixel_sharpen_enabled or self.settings.pixel_sharpen_percent <= 0:
            return image
        return image.filter(
            ImageFilter.UnsharpMask(
                radius=self.settings.pixel_sharpen_radius,
                percent=self.settings.pixel_sharpen_percent,
                threshold=self.settings.pixel_sharpen_threshold,
            )
        )

    def _realesrgan_upscale(self, image: Image.Image, *, width: int, height: int) -> Optional[Image.Image]:
        model_path = self.settings.realesrgan_model_path.strip()
        if not model_path:
            return None

        import numpy as np
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        upsampler = self._get_upsampler(RealESRGANer, RRDBNet, model_path)
        scale = max(width / image.width, height / image.height)
        outscale = max(1.0, min(4.0, scale))
        output, _ = upsampler.enhance(np.array(image), outscale=outscale)
        return Image.fromarray(output).resize((width, height), Image.Resampling.LANCZOS)

    def _get_upsampler(self, realesrganer, rrdbnet, model_path: str):
        if self._upsampler is not None:
            return self._upsampler

        model = rrdbnet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self._upsampler = realesrganer(
            scale=4,
            model_path=str(Path(model_path)),
            model=model,
            tile=self.settings.realesrgan_tile,
            tile_pad=self.settings.realesrgan_tile_pad,
            pre_pad=self.settings.realesrgan_pre_pad,
            half=self.settings.torch_dtype in {"float16", "bfloat16"},
            device=None,
        )
        return self._upsampler


@lru_cache
def realesrgan_available() -> bool:
    try:
        import basicsr  # noqa: F401
        import realesrgan  # noqa: F401
    except Exception:
        return False
    return True
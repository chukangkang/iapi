import asyncio
import inspect
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
        model_path = self._resolve_realesrgan_model_path()
        if model_path is None:
            return None

        import numpy as np
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.archs.srvgg_arch import SRVGGNetCompact

        upsampler = self._get_upsampler(RealESRGANer, RRDBNet, SRVGGNetCompact, model_path)
        upscaled = image
        for _ in range(self.settings.realesrgan_max_passes):
            output, _ = upsampler.enhance(np.array(upscaled), outscale=4)
            upscaled = Image.fromarray(output).convert("RGB")
            if upscaled.width >= width and upscaled.height >= height:
                break
        return upscaled.resize((width, height), Image.Resampling.LANCZOS)

    def _resolve_realesrgan_model_path(self) -> Optional[Path]:
        raw_model_path = self.settings.realesrgan_model_path.strip()
        if not raw_model_path:
            return None

        model_path = Path(raw_model_path)
        if model_path.is_dir():
            model_path = model_path / self.settings.realesrgan_model_name

        if model_path.suffix.lower() == ".safetensors":
            raise ValueError(
                "Real-ESRGAN Python inference expects a .pth checkpoint. "
                f"Configured file is {model_path}. Use RealESRGAN_x4plus.pth instead."
            )
        if model_path.suffix.lower() != ".pth":
            raise ValueError(f"REALESRGAN_MODEL_PATH must point to a .pth checkpoint, got: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Real-ESRGAN checkpoint not found: {model_path}")
        return model_path

    def _get_upsampler(self, realesrganer, rrdbnet, srvggnet, model_path: Path):
        if self._upsampler is not None:
            return self._upsampler

        model, model_paths = self._build_model(rrdbnet, srvggnet, model_path)
        upsampler_kwargs = {
            "scale": 4,
            "model_path": model_paths,
            "model": model,
            "tile": self.settings.realesrgan_tile,
            "tile_pad": self.settings.realesrgan_tile_pad,
            "pre_pad": self.settings.realesrgan_pre_pad,
            "half": self.settings.torch_dtype in {"float16", "bfloat16"},
            "device": None,
        }
        if "dni_weight" in inspect.signature(realesrganer.__init__).parameters:
            upsampler_kwargs["dni_weight"] = self._dni_weight(model_paths)

        self._upsampler = realesrganer(**upsampler_kwargs)
        return self._upsampler

    def _dni_weight(self, model_paths):
        if isinstance(model_paths, list):
            denoise = self.settings.realesrgan_denoise_strength
            return [denoise, 1 - denoise]
        return None

    def _build_model(self, rrdbnet, srvggnet, model_path: Path):
        model_name = model_path.name.lower()
        if model_name in {"realesr-general-x4v3.pth", "realesr-general-wdn-x4v3.pth"}:
            model = srvggnet(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
            if model_name == "realesr-general-x4v3.pth":
                wdn_path = model_path.with_name("realesr-general-wdn-x4v3.pth")
                if wdn_path.exists() and self.settings.realesrgan_denoise_strength < 1.0:
                    return model, [str(model_path), str(wdn_path)]
            return model, str(model_path)

        return rrdbnet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), str(model_path)


@lru_cache
def realesrgan_available() -> bool:
    try:
        import basicsr  # noqa: F401
        import realesrgan  # noqa: F401
    except Exception:
        return False
    return True
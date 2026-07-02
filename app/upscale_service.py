import asyncio
import inspect
import logging
import sys
import types
import contextlib
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFilter

from app.config import Settings


logger = logging.getLogger(__name__)


REALESRGAN_RELEASE_BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@dataclass(frozen=True)
class RealESRGANModelSpec:
    name: str
    architecture: str
    scale: int
    url: str
    denoise_url: Optional[str] = None


REALESRGAN_MODELS = {
    "realesrgan_x4plus": RealESRGANModelSpec(
        name="RealESRGAN_x4plus",
        architecture="rrdbnet_x4plus",
        scale=4,
        url=f"{REALESRGAN_RELEASE_BASE}/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    "realesrnet_x4plus": RealESRGANModelSpec(
        name="RealESRNet_x4plus",
        architecture="rrdbnet_x4plus",
        scale=4,
        url=f"{REALESRGAN_RELEASE_BASE}/v0.1.1/RealESRNet_x4plus.pth",
    ),
    "realesrgan_x4plus_anime_6b": RealESRGANModelSpec(
        name="RealESRGAN_x4plus_anime_6B",
        architecture="rrdbnet_x4plus_anime_6b",
        scale=4,
        url=f"{REALESRGAN_RELEASE_BASE}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
    "realesrgan_x2plus": RealESRGANModelSpec(
        name="RealESRGAN_x2plus",
        architecture="rrdbnet_x2plus",
        scale=2,
        url=f"{REALESRGAN_RELEASE_BASE}/v0.2.1/RealESRGAN_x2plus.pth",
    ),
    "realesr-animevideov3": RealESRGANModelSpec(
        name="realesr-animevideov3",
        architecture="srvgg_animevideov3",
        scale=4,
        url=f"{REALESRGAN_RELEASE_BASE}/v0.2.5.0/realesr-animevideov3.pth",
    ),
    "realesr-general-x4v3": RealESRGANModelSpec(
        name="realesr-general-x4v3",
        architecture="srvgg_general_x4v3",
        scale=4,
        url=f"{REALESRGAN_RELEASE_BASE}/v0.2.5.0/realesr-general-x4v3.pth",
        denoise_url=f"{REALESRGAN_RELEASE_BASE}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    ),
    "realesr-general-wdn-x4v3": RealESRGANModelSpec(
        name="realesr-general-wdn-x4v3",
        architecture="srvgg_general_x4v3",
        scale=4,
        url=f"{REALESRGAN_RELEASE_BASE}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    ),
}

REALESRGAN_MODEL_ALIASES = {
    "realesrgan-x4plus": "realesrgan_x4plus",
    "realesrnet-x4plus": "realesrnet_x4plus",
    "realesrgan-x4plus-anime": "realesrgan_x4plus_anime_6b",
    "realesrgan-x4plus-anime-6b": "realesrgan_x4plus_anime_6b",
    "realesr-animevideov3-x2": "realesr-animevideov3",
    "realesr-animevideov3-x3": "realesr-animevideov3",
}


class ImageUpscaleService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._upsampler = None
        self._upsampler_key = None
        self._face_enhancer = None
        self._face_enhancer_key = None

    async def upscale(
        self,
        image: Image.Image,
        *,
        width: int,
        height: int,
        method: str,
        fit_mode: str,
        face_enhance: Optional[bool] = None,
    ) -> Image.Image:
        return await asyncio.to_thread(
            self._upscale_sync,
            image,
            width=width,
            height=height,
            method=method,
            fit_mode=fit_mode,
            face_enhance=face_enhance,
        )

    async def prepare(self, *, method: str) -> None:
        await asyncio.to_thread(self._prepare_sync, method=method)

    def _prepare_sync(self, *, method: str) -> None:
        if method != "realesrgan":
            return
        model_path, spec = self._resolve_realesrgan_model()

        _patch_torchvision_functional_tensor()

        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        from basicsr.utils.download_util import load_file_from_url

        upsampler = self._get_upsampler(RealESRGANer, RRDBNet, SRVGGNetCompact, load_file_from_url, model_path, spec)
        if self.settings.realesrgan_face_enhance:
            from gfpgan import GFPGANer

            self._get_face_enhancer(GFPGANer, upsampler)

    def _upscale_sync(self, image: Image.Image, *, width: int, height: int, method: str, fit_mode: str, face_enhance: Optional[bool] = None) -> Image.Image:
        image = image.convert("RGB")
        if method == "realesrgan":
            upscaled = self._realesrgan_upscale(image, width=width, height=height, fit_mode=fit_mode, face_enhance=face_enhance)
            if upscaled is not None:
                return upscaled
        upscaled = self._fit_to_target(image, width=width, height=height, fit_mode=fit_mode)
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

    def _realesrgan_upscale(self, image: Image.Image, *, width: int, height: int, fit_mode: str, face_enhance: Optional[bool] = None) -> Optional[Image.Image]:
        model_path, spec = self._resolve_realesrgan_model()

        _patch_torchvision_functional_tensor()

        import numpy as np
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        from basicsr.utils.download_util import load_file_from_url

        upsampler = self._get_upsampler(RealESRGANer, RRDBNet, SRVGGNetCompact, load_file_from_url, model_path, spec)
        upscaled = image
        outscale = self._resolve_outscale(spec)
        enhance_kwargs = {"outscale": outscale}
        if "alpha_upsampler" in inspect.signature(upsampler.enhance).parameters:
            enhance_kwargs["alpha_upsampler"] = self.settings.realesrgan_alpha_upsampler
        if face_enhance if face_enhance is not None else self.settings.realesrgan_face_enhance:
            from gfpgan import GFPGANer

            face_enhancer = self._get_face_enhancer(GFPGANer, upsampler)
            for _ in range(self.settings.realesrgan_max_passes):
                with contextlib.redirect_stdout(io.StringIO()):
                    _, _, output = face_enhancer.enhance(
                        np.array(upscaled),
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                    )
                upscaled = Image.fromarray(output).convert("RGB")
                if upscaled.width >= width and upscaled.height >= height:
                    break
            return self._fit_to_target(upscaled, width=width, height=height, fit_mode=fit_mode)
        for _ in range(self.settings.realesrgan_max_passes):
            with contextlib.redirect_stdout(io.StringIO()):
                output, _ = upsampler.enhance(np.array(upscaled), **enhance_kwargs)
            upscaled = Image.fromarray(output).convert("RGB")
            if upscaled.width >= width and upscaled.height >= height:
                break
        return self._fit_to_target(upscaled, width=width, height=height, fit_mode=fit_mode)

    def _fit_to_target(self, image: Image.Image, *, width: int, height: int, fit_mode: str) -> Image.Image:
        if fit_mode == "stretch":
            return image.resize((width, height), Image.Resampling.LANCZOS)

        source_ratio = image.width / image.height
        target_ratio = width / height
        if fit_mode == "cover":
            if source_ratio > target_ratio:
                resized_height = height
                resized_width = int(round(height * source_ratio))
            else:
                resized_width = width
                resized_height = int(round(width / source_ratio))
            resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
            left = max(0, (resized_width - width) // 2)
            top = max(0, (resized_height - height) // 2)
            return resized.crop((left, top, left + width, top + height))

        if source_ratio > target_ratio:
            resized_width = width
            resized_height = int(round(width / source_ratio))
        else:
            resized_height = height
            resized_width = int(round(height * source_ratio))
        resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), self.settings.upscale_fill_color)
        canvas.paste(resized, ((width - resized_width) // 2, (height - resized_height) // 2))
        return canvas

    def _resolve_realesrgan_model(self) -> tuple[Path, RealESRGANModelSpec]:
        spec = self._resolve_realesrgan_model_spec()
        raw_model_path = self.settings.realesrgan_model_path.strip()
        model_path = Path(raw_model_path) if raw_model_path else Path("weights")

        if model_path.is_dir() or not model_path.suffix:
            model_path = model_path / f"{spec.name}.pth"

        if model_path.suffix.lower() == ".safetensors":
            raise ValueError(
                "Real-ESRGAN Python inference expects a .pth checkpoint. "
                f"Configured file is {model_path}. Use RealESRGAN_x4plus.pth instead."
            )
        if model_path.suffix.lower() != ".pth":
            raise ValueError(f"REALESRGAN_MODEL_PATH must point to a .pth checkpoint, got: {model_path}")
        return model_path, spec

    def _resolve_realesrgan_model_spec(self) -> RealESRGANModelSpec:
        raw_name = self.settings.realesrgan_model_name.strip() or "realesr-general-x4v3"
        normalized = _normalize_realesrgan_model_name(raw_name)
        spec = REALESRGAN_MODELS.get(normalized)
        if spec is not None:
            return spec
        supported = ", ".join(sorted(spec.name for spec in REALESRGAN_MODELS.values()))
        raise ValueError(f"Unsupported REALESRGAN_MODEL_NAME={raw_name!r}. Supported models: {supported}")

    def _get_upsampler(self, realesrganer, rrdbnet, srvggnet, load_file_from_url, model_path: Path, spec: RealESRGANModelSpec):
        model_path, model_paths = self._ensure_model_paths(model_path, spec, load_file_from_url)
        upsampler_key = (
            tuple(model_paths) if isinstance(model_paths, list) else model_paths,
            self.settings.realesrgan_denoise_strength,
            self.settings.realesrgan_tile,
            self.settings.realesrgan_tile_pad,
            self.settings.realesrgan_pre_pad,
            self.settings.realesrgan_fp32,
            self.settings.realesrgan_gpu_id,
        )
        if self._upsampler is not None and self._upsampler_key == upsampler_key:
            return self._upsampler

        model = self._build_model(rrdbnet, srvggnet, spec)
        upsampler_kwargs = {
            "scale": spec.scale,
            "model_path": model_paths,
            "model": model,
            "tile": self.settings.realesrgan_tile,
            "tile_pad": self.settings.realesrgan_tile_pad,
            "pre_pad": self.settings.realesrgan_pre_pad,
            "half": not self.settings.realesrgan_fp32 and self.settings.torch_dtype in {"float16", "bfloat16"},
        }
        init_params = inspect.signature(realesrganer.__init__).parameters
        if "device" in init_params:
            upsampler_kwargs["device"] = None
        elif "gpu_id" in init_params:
            upsampler_kwargs["gpu_id"] = self.settings.realesrgan_gpu_id
        if "dni_weight" in init_params:
            upsampler_kwargs["dni_weight"] = self._dni_weight(model_paths)

        self._upsampler = realesrganer(**upsampler_kwargs)
        self._upsampler_key = upsampler_key
        return self._upsampler

    def _get_face_enhancer(self, gfpganer, upsampler):
        face_enhancer_key = (
            id(upsampler),
            self._resolve_outscale(self._resolve_realesrgan_model_spec()),
        )
        if self._face_enhancer is not None and self._face_enhancer_key == face_enhancer_key:
            return self._face_enhancer
        self._face_enhancer = gfpganer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=self._resolve_outscale(self._resolve_realesrgan_model_spec()),
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )
        self._face_enhancer_key = face_enhancer_key
        return self._face_enhancer

    def _ensure_model_paths(self, model_path: Path, spec: RealESRGANModelSpec, load_file_from_url):
        model_path = self._ensure_model_file(model_path, spec.url, load_file_from_url)
        if spec.name == "realesr-general-x4v3" and self.settings.realesrgan_denoise_strength < 1.0:
            wdn_path = model_path.with_name("realesr-general-wdn-x4v3.pth")
            if spec.denoise_url is not None:
                wdn_path = self._ensure_model_file(wdn_path, spec.denoise_url, load_file_from_url)
            return model_path, [str(model_path), str(wdn_path)]
        return model_path, str(model_path)

    def _ensure_model_file(self, model_path: Path, url: str, load_file_from_url) -> Path:
        if model_path.exists():
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Real-ESRGAN model from %s to %s", url, model_path.parent)
        downloaded_path = Path(load_file_from_url(url=url, model_dir=str(model_path.parent), progress=True, file_name=model_path.name))
        if not downloaded_path.exists():
            raise FileNotFoundError(f"Real-ESRGAN checkpoint download failed: {downloaded_path}")
        return downloaded_path

    def _dni_weight(self, model_paths):
        if isinstance(model_paths, list):
            denoise = self.settings.realesrgan_denoise_strength
            return [denoise, 1 - denoise]
        return None

    def _resolve_outscale(self, spec: RealESRGANModelSpec) -> float:
        if self.settings.realesrgan_outscale > 0:
            return self.settings.realesrgan_outscale
        return float(spec.scale)

    def _build_model(self, rrdbnet, srvggnet, spec: RealESRGANModelSpec):
        if spec.architecture == "srvgg_general_x4v3":
            return srvggnet(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
        if spec.architecture == "srvgg_animevideov3":
            return srvggnet(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
        if spec.architecture == "rrdbnet_x4plus_anime_6b":
            return rrdbnet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        if spec.architecture == "rrdbnet_x2plus":
            return rrdbnet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        return rrdbnet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)


def _normalize_realesrgan_model_name(model_name: str) -> str:
    normalized = Path(model_name).stem.strip().lower()
    normalized = REALESRGAN_MODEL_ALIASES.get(normalized, normalized)
    return normalized


@lru_cache
def realesrgan_available() -> bool:
    return realesrgan_import_error() is None


@lru_cache
def realesrgan_import_error() -> Optional[str]:
    try:
        _patch_torchvision_functional_tensor()
        import basicsr  # noqa: F401
        import realesrgan  # noqa: F401
    except Exception:
        import traceback

        return traceback.format_exc(limit=5)
    return None


def _patch_torchvision_functional_tensor() -> None:
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        from torchvision.transforms.functional import rgb_to_grayscale
    except Exception:
        return

    module = types.ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = module
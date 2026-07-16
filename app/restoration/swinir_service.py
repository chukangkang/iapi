import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from PIL import Image

from app.config import Settings
from app.restoration.analyzer import DegradationReport


logger = logging.getLogger(__name__)

SWINIR_RELEASE_BASE = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0"


@dataclass(frozen=True)
class SwinIRModelSpec:
    task: str
    filename: str
    url: str
    window_size: int
    in_chans: int
    img_size: int
    embed_dim: int
    depths: tuple[int, ...]
    num_heads: tuple[int, ...]
    mlp_ratio: float
    resi_connection: str


def _denoise_spec(level: int) -> SwinIRModelSpec:
    filename = f"003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-M_noise{level}.pth"
    return SwinIRModelSpec(
        task="denoise",
        filename=filename,
        url=f"{SWINIR_RELEASE_BASE}/{filename}",
        window_size=8,
        in_chans=3,
        img_size=64,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        mlp_ratio=2.0,
        resi_connection="1conv",
    )


def _jpeg_spec(quality: int) -> SwinIRModelSpec:
    filename = f"006_CAR_DFWB_s126w7_SwinIR-M_jpeg{quality}.pth"
    return SwinIRModelSpec(
        task="jpeg",
        filename=filename,
        url=f"{SWINIR_RELEASE_BASE}/{filename}",
        window_size=7,
        in_chans=1,
        img_size=126,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        mlp_ratio=2.0,
        resi_connection="1conv",
    )


class SwinIRService:
    """SwinIR denoising/JPEG restoration with lazy model loading and tiled inference."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._model_key: Optional[tuple[Path, SwinIRModelSpec, str]] = None

    async def restore(self, image: Image.Image, *, report: DegradationReport) -> Image.Image:
        if not self.settings.swinir_enabled:
            return image
        try:
            return await asyncio.to_thread(self._restore_sync, image, report)
        except Exception as exc:
            logger.warning("SwinIR failed; continuing with the original image: %s", exc)
            return image

    def select_model(self, report: DegradationReport) -> SwinIRModelSpec:
        if report.blockiness_score >= 0.25 and report.blockiness_score >= report.noise_score:
            quality = 20 if report.blockiness_score >= 0.6 else 40
            return _jpeg_spec(quality)
        level = 50 if report.noise_score >= 0.4 else 25 if report.noise_score >= 0.2 else 15
        return _denoise_spec(level)

    def resolve_model_path(self, spec: SwinIRModelSpec) -> Path:
        configured = self.settings.swinir_model_path.strip()
        if configured:
            configured_path = Path(configured)
            if configured_path.suffix:
                if not configured_path.is_file():
                    raise FileNotFoundError(f"SWINIR_MODEL_PATH does not exist: {configured_path}")
                return configured_path
            model_path = configured_path / spec.filename
        else:
            model_path = Path("weights") / "SwinIR" / spec.filename
        if model_path.is_file():
            return model_path
        if not self.settings.swinir_auto_download:
            raise FileNotFoundError(f"SwinIR checkpoint does not exist: {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading SwinIR model from %s to %s", spec.url, model_path)
        temporary_path = model_path.with_suffix(model_path.suffix + ".part")
        try:
            urlretrieve(spec.url, temporary_path)
            temporary_path.replace(model_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return model_path

    def _restore_sync(self, image: Image.Image, report: DegradationReport) -> Image.Image:
        import numpy as np
        import torch

        spec = self.select_model(report)
        model_path = self.resolve_model_path(spec)
        device = self._resolve_device(torch)
        model = self._get_model(torch, model_path, spec, device)

        original = image.convert("RGB")
        if spec.in_chans == 1:
            ycbcr = original.convert("YCbCr")
            y, cb, cr = ycbcr.split()
            source = np.asarray(y, dtype=np.float32)[..., None] / 255.0
        else:
            ycbcr = None
            source = np.asarray(original, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(source.transpose(2, 0, 1)).unsqueeze(0).to(device)
        tensor = tensor.float() if self.settings.swinir_fp32 or device.type == "cpu" else tensor.half()
        restored = self._infer_tiled(torch, model, tensor, spec.window_size)
        output = restored.squeeze(0).float().clamp_(0, 1).cpu().numpy().transpose(1, 2, 0)
        output = (output * 255.0).round().astype("uint8")
        if spec.in_chans == 1 and ycbcr is not None:
            restored_y = Image.fromarray(output[..., 0], mode="L")
            return Image.merge("YCbCr", (restored_y, cb, cr)).convert("RGB")
        return Image.fromarray(output, mode="RGB")

    def _resolve_device(self, torch):
        if self.settings.device == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        index = self.settings.swinir_gpu_id
        return torch.device("cuda" if index is None else f"cuda:{index}")

    def _get_model(self, torch, model_path: Path, spec: SwinIRModelSpec, device):
        key = (model_path.resolve(), spec, str(device))
        if self._model is not None and self._model_key == key:
            return self._model
        from basicsr.archs.swinir_arch import SwinIR

        model = SwinIR(
            upscale=1,
            in_chans=spec.in_chans,
            img_size=spec.img_size,
            window_size=spec.window_size,
            img_range=1.0,
            depths=list(spec.depths),
            embed_dim=spec.embed_dim,
            num_heads=list(spec.num_heads),
            mlp_ratio=spec.mlp_ratio,
            upsampler="",
            resi_connection=spec.resi_connection,
        )
        checkpoint = torch.load(str(model_path), map_location="cpu")
        state_dict = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)
        if not self.settings.swinir_fp32 and device.type == "cuda":
            model.half()
        self._model = model
        self._model_key = key
        return model

    def _infer_tiled(self, torch, model, image, window_size: int):
        height, width = image.shape[-2:]
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        padded = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        tile = self.settings.swinir_tile
        with torch.inference_mode():
            if tile <= 0 or (padded.shape[-2] <= tile and padded.shape[-1] <= tile):
                output = model(padded)
            else:
                output = self._tile_forward(torch, model, padded, tile, window_size)
        return output[..., :height, :width]

    def _tile_forward(self, torch, model, image, tile: int, window_size: int):
        tile = max(window_size, tile - tile % window_size)
        overlap = min(self.settings.swinir_tile_overlap, tile - window_size)
        stride = max(window_size, tile - overlap)
        output = torch.zeros_like(image)
        weight = torch.zeros_like(image)
        height, width = image.shape[-2:]
        for y in range(0, height, stride):
            y = min(y, max(0, height - tile))
            for x in range(0, width, stride):
                x = min(x, max(0, width - tile))
                patch = image[..., y : y + tile, x : x + tile]
                restored = model(patch)
                output[..., y : y + patch.shape[-2], x : x + patch.shape[-1]] += restored
                weight[..., y : y + patch.shape[-2], x : x + patch.shape[-1]] += 1
        return output / weight.clamp_min_(1)
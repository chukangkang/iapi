import asyncio
from typing import Optional

from PIL import Image

from app.config import Settings


class FluxImageService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self._lock = asyncio.Lock()
        self._device = None
        self._dtype = None

    async def generate(
        self,
        *,
        prompt: str,
        image: Optional[Image.Image],
        width: int,
        height: int,
        num_inference_steps: int,
        seed: Optional[int],
    ) -> Image.Image:
        async with self._lock:
            return await asyncio.to_thread(
                self._generate_sync,
                prompt=prompt,
                image=image,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )

    def _generate_sync(
        self,
        *,
        prompt: str,
        image: Optional[Image.Image],
        width: int,
        height: int,
        num_inference_steps: int,
        seed: Optional[int],
    ) -> Image.Image:
        import torch

        pipe = self._get_pipeline()
        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
        }
        if image is not None:
            kwargs["image"] = image
        if seed is not None:
            kwargs["generator"] = torch.Generator(device=self._generator_device()).manual_seed(seed)

        with torch.inference_mode():
            result = pipe(**kwargs)
        return result.images[0]

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from diffusers import Flux2KleinKVPipeline

        self._device = self._resolve_device(torch)
        self._dtype = self._resolve_dtype(torch)

        load_kwargs = {}
        if self._dtype is not None:
            load_kwargs["torch_dtype"] = self._dtype
        if self.settings.hf_token and not self.settings.hf_token.startswith("replace-with"):
            load_kwargs["token"] = self.settings.hf_token

        pipe = Flux2KleinKVPipeline.from_pretrained(self.settings.model_path, **load_kwargs)

        if self.settings.enable_cpu_offload and hasattr(pipe, "enable_model_cpu_offload") and self._device.startswith("cuda"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self._device)

        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        self._pipe = pipe
        return pipe

    def _resolve_device(self, torch) -> str:
        if self.settings.device != "auto":
            return self.settings.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, torch):
        if self.settings.torch_dtype == "auto":
            return None
        return getattr(torch, self.settings.torch_dtype)

    def _generator_device(self) -> str:
        if not self._device or self._device == "mps":
            return "cpu"
        return self._device

import asyncio
import gc
import inspect
import logging
from typing import Optional

from PIL import Image

from app.config import Settings
from app.pipeline_utils import apply_pipeline_cpu_offload, apply_pipeline_memory_settings, get_pipeline_device_map_kwargs, uses_pipeline_device_map


logger = logging.getLogger(__name__)


class FluxImageService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self._device = None
        self._dtype = None
        self._cpu_offload_enabled = False

    async def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        image: Optional[Image.Image],
        width: int,
        height: int,
        num_inference_steps: int,
        seed: Optional[int],
        strength: Optional[float] = None,
    ) -> Image.Image:
        return await asyncio.to_thread(
            self._generate_sync,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            strength=strength,
        )

    async def prepare(self) -> None:
        await asyncio.to_thread(self._get_pipeline)

    def _generate_sync(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        image: Optional[Image.Image],
        width: int,
        height: int,
        num_inference_steps: int,
        seed: Optional[int],
        strength: Optional[float],
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
        if negative_prompt and "negative_prompt" in inspect.signature(pipe.__call__).parameters:
            kwargs["negative_prompt"] = negative_prompt
        if strength is not None and "strength" in inspect.signature(pipe.__call__).parameters:
            kwargs["strength"] = strength
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
        load_kwargs.update(get_pipeline_device_map_kwargs(self.settings, torch, self._device))

        pipe = Flux2KleinKVPipeline.from_pretrained(self.settings.model_path, **load_kwargs)

        device_map_enabled = uses_pipeline_device_map(load_kwargs)
        self._cpu_offload_enabled = False if device_map_enabled else apply_pipeline_cpu_offload(pipe, self.settings, self._device)
        if not self._cpu_offload_enabled:
            if not device_map_enabled:
                pipe.to(self._device)

        apply_pipeline_memory_settings(pipe, self.settings)

        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        self._pipe = pipe
        return pipe

    def unload(self) -> None:
        if self._pipe is None:
            return
        try:
            if not self._cpu_offload_enabled and not self._has_device_map() and hasattr(self._pipe, "to"):
                self._pipe.to("cpu")
                logger.info("Moved FLUX pipeline to CPU before release")
            else:
                logger.info("Releasing FLUX pipeline")
        except Exception as exc:
            logger.warning("Failed to unload FLUX pipeline: %s", exc)
        self._pipe = None
        self._cpu_offload_enabled = False
        self._release_torch_memory()

    def _release_torch_memory(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:
            logger.debug("Failed to release torch cache: %s", exc)

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

    def _has_device_map(self) -> bool:
        return bool(getattr(self._pipe, "hf_device_map", None))

    def _generator_device(self) -> str:
        if not self._device or self._device == "mps":
            return "cpu"
        return self._device

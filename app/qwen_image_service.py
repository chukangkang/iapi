import asyncio
import inspect
import logging
import re
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from app.config import Settings


logger = logging.getLogger(__name__)


class QwenImageService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self._device = None
        self._dtype = None
        self._lora_loaded = False

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
        )

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
    ) -> Image.Image:
        import torch

        pipe = self._get_pipeline()
        signature = inspect.signature(pipe.__call__).parameters
        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
        }
        if image is not None:
            image_argument = self._image_argument_name(signature)
            if image_argument is None:
                raise RuntimeError(f"{self.settings.qwen_image_pipeline_class} does not support image input for /v1/images/edits.")
            kwargs[image_argument] = self._prepare_image(image, width, height)
        if negative_prompt and "negative_prompt" in signature:
            kwargs["negative_prompt"] = negative_prompt
        elif "negative_prompt" in signature:
            kwargs["negative_prompt"] = " "
        if "guidance_scale" in signature:
            kwargs["guidance_scale"] = self.settings.qwen_image_guidance_scale
        if "true_cfg_scale" in signature:
            kwargs["true_cfg_scale"] = self.settings.qwen_image_true_cfg_scale
        if seed is not None:
            kwargs["generator"] = torch.Generator(device=self._generator_device()).manual_seed(seed)

        logger.info(
            "Qwen Image generation: size=%sx%s steps=%s seed=%s guidance_scale=%s true_cfg_scale=%s lora_loaded=%s",
            width,
            height,
            num_inference_steps,
            seed,
            kwargs.get("guidance_scale"),
            kwargs.get("true_cfg_scale"),
            self._lora_loaded,
        )

        with torch.inference_mode():
            result = pipe(**kwargs)
        return result.images[0].convert("RGB")

    def _image_argument_name(self, signature) -> Optional[str]:
        for name in ("image", "images", "input_image", "init_image"):
            if name in signature:
                return name
        return None

    def _prepare_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        if self.settings.qwen_edit_input_fit_mode == "cover":
            return ImageOps.fit(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        return ImageOps.pad(
            image.convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
            color=self.settings.qwen_edit_background_color,
            centering=(0.5, 0.5),
        )

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        import diffusers

        pipeline_cls = getattr(diffusers, self.settings.qwen_image_pipeline_class)
        self._device = self._resolve_device(torch)
        self._dtype = self._resolve_dtype(torch)

        load_kwargs = {}
        if self._dtype is not None:
            load_kwargs["torch_dtype"] = self._dtype
        if self.settings.hf_token and not self.settings.hf_token.startswith("replace-with"):
            load_kwargs["token"] = self.settings.hf_token

        model_path = self.settings.qwen_image_model_path
        if self._looks_like_single_file(model_path) and hasattr(pipeline_cls, "from_single_file"):
            pipe = pipeline_cls.from_single_file(model_path, **load_kwargs)
        else:
            pipe = pipeline_cls.from_pretrained(model_path, **load_kwargs)

        if self.settings.enable_cpu_offload and hasattr(pipe, "enable_model_cpu_offload") and self._device.startswith("cuda"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self._device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        self._load_lora(pipe)
        self._apply_scheduler_config(pipe)
        self._pipe = pipe
        return pipe

    def _load_lora(self, pipe) -> None:
        if self._lora_loaded or not self.settings.qwen_image_lora_path:
            return
        configured_adapter_name = self.settings.qwen_image_lora_adapter_name
        adapter_name = self._safe_adapter_name(configured_adapter_name)
        kwargs = {"adapter_name": adapter_name}
        weight_name = self.settings.qwen_image_lora_weight_name or self._weight_name_from_adapter_name(configured_adapter_name)
        if weight_name:
            kwargs["weight_name"] = weight_name
        pipe.load_lora_weights(self.settings.qwen_image_lora_path, **kwargs)
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters([adapter_name], adapter_weights=[self.settings.qwen_image_lora_scale])
        elif hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora(lora_scale=self.settings.qwen_image_lora_scale)
        self._lora_loaded = True
        logger.info(
            "Loaded Qwen Image LoRA: path=%s weight_name=%s adapter_name=%s scale=%s",
            self.settings.qwen_image_lora_path,
            weight_name or "<auto>",
            adapter_name,
            self.settings.qwen_image_lora_scale,
        )

    def _apply_scheduler_config(self, pipe) -> None:
        scheduler = getattr(pipe, "scheduler", None)
        if scheduler is None:
            return
        scheduler_config = {
            "exponential_shift_mu": self.settings.qwen_image_scheduler_exponential_shift_mu,
            "use_dynamic_shifting": self.settings.qwen_image_scheduler_use_dynamic_shifting,
            "shift_terminal": self.settings.qwen_image_scheduler_shift_terminal,
        }
        if hasattr(scheduler, "from_config"):
            pipe.scheduler = scheduler.__class__.from_config(scheduler.config, **scheduler_config)
        else:
            for key, value in scheduler_config.items():
                setattr(scheduler, key, value)
        logger.info("Applied Qwen Image scheduler config: %s", scheduler_config)

    def _safe_adapter_name(self, adapter_name: str) -> str:
        value = (adapter_name or "qwen_image_2512_turbo").strip()
        value = Path(value).stem if Path(value).suffix else value
        value = re.sub(r"[^0-9A-Za-z_]", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        return value or "qwen_image_2512_turbo"

    def _weight_name_from_adapter_name(self, adapter_name: str) -> str:
        value = (adapter_name or "").strip()
        return Path(value).name if Path(value).suffix.lower() == ".safetensors" else ""

    def _looks_like_single_file(self, model_path: str) -> bool:
        suffix = Path(model_path).suffix.lower()
        return suffix in {".gguf", ".safetensors", ".ckpt", ".pt", ".pth"}

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

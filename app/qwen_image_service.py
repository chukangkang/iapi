import asyncio
import inspect
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from app.config import Settings


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
        self._pipe = pipe
        return pipe

    def _load_lora(self, pipe) -> None:
        if self._lora_loaded or not self.settings.qwen_image_lora_path:
            return
        kwargs = {"adapter_name": self.settings.qwen_image_lora_adapter_name}
        if self.settings.qwen_image_lora_weight_name:
            kwargs["weight_name"] = self.settings.qwen_image_lora_weight_name
        pipe.load_lora_weights(self.settings.qwen_image_lora_path, **kwargs)
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters([self.settings.qwen_image_lora_adapter_name], adapter_weights=[self.settings.qwen_image_lora_scale])
        elif hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora(lora_scale=self.settings.qwen_image_lora_scale)
        self._lora_loaded = True

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

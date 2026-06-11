import asyncio
import inspect
from typing import Optional

from PIL import Image

from app.config import Settings


class QwenImageEditService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self._device = None
        self._dtype = None

    async def edit(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        image: Image.Image,
        width: int,
        height: int,
        num_inference_steps: int,
        seed: Optional[int],
        guidance_scale: float,
        strength: Optional[float],
    ) -> Image.Image:
        return await asyncio.to_thread(
            self._edit_sync,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            guidance_scale=guidance_scale,
            strength=strength,
        )

    def _edit_sync(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        image: Image.Image,
        width: int,
        height: int,
        num_inference_steps: int,
        seed: Optional[int],
        guidance_scale: float,
        strength: Optional[float],
    ) -> Image.Image:
        import torch

        pipe = self._get_pipeline()
        signature = inspect.signature(pipe.__call__).parameters
        kwargs = {
            "prompt": prompt,
            "image": image.convert("RGB"),
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
        }
        if negative_prompt and "negative_prompt" in signature:
            kwargs["negative_prompt"] = negative_prompt
        elif "negative_prompt" in signature:
            kwargs["negative_prompt"] = " "
        if "guidance_scale" in signature:
            kwargs["guidance_scale"] = guidance_scale
        if "true_cfg_scale" in signature:
            kwargs["true_cfg_scale"] = self.settings.qwen_edit_true_cfg_scale
        if strength is not None and "strength" in signature:
            kwargs["strength"] = strength
        if seed is not None:
            kwargs["generator"] = torch.Generator(device=self._generator_device()).manual_seed(seed)

        with torch.inference_mode():
            result = pipe(**kwargs)
        return result.images[0].convert("RGB")

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        import diffusers

        pipeline_cls = getattr(diffusers, self.settings.qwen_edit_pipeline_class)
        self._device = self._resolve_device(torch)
        self._dtype = self._resolve_dtype(torch)

        load_kwargs = {}
        if self._dtype is not None:
            load_kwargs["torch_dtype"] = self._dtype
        if self.settings.hf_token and not self.settings.hf_token.startswith("replace-with"):
            load_kwargs["token"] = self.settings.hf_token
        quantization_config = self._quantization_config(diffusers)
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = self.settings.qwen_edit_device_map

        pipe = pipeline_cls.from_pretrained(self.settings.qwen_edit_model_path, **load_kwargs)
        if quantization_config is not None:
            pass
        elif self.settings.enable_cpu_offload and hasattr(pipe, "enable_model_cpu_offload") and self._device.startswith("cuda"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self._device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        self._pipe = pipe
        return pipe

    def _quantization_config(self, diffusers):
        if self.settings.qwen_edit_quantization == "none":
            return None
        pipeline_quantization_config = getattr(diffusers, "PipelineQuantizationConfig", None)
        if pipeline_quantization_config is not None:
            if self.settings.qwen_edit_quantization == "8bit":
                return pipeline_quantization_config(
                    quant_backend="bitsandbytes_8bit",
                    quant_kwargs={"load_in_8bit": True, "llm_int8_enable_fp32_cpu_offload": True},
                )
            return pipeline_quantization_config(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": self._dtype},
            )

        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError("QWEN_EDIT_QUANTIZATION requires bitsandbytes and a recent diffusers/transformers version.") from exc

        if self.settings.qwen_edit_quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=self._dtype)

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
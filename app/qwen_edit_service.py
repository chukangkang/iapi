import asyncio
import inspect
from typing import Optional

from PIL import Image, ImageOps

from app.config import Settings


class QwenImageEditService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self._device = None
        self._dtype = None
        self._active_adapter = None

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
        lora_path: Optional[str] = None,
        lora_weight_name: Optional[str] = None,
        lora_scale: float = 1.0,
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
            lora_path=lora_path,
            lora_weight_name=lora_weight_name,
            lora_scale=lora_scale,
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
        lora_path: Optional[str],
        lora_weight_name: Optional[str],
        lora_scale: float,
    ) -> Image.Image:
        import torch

        pipe = self._get_pipeline()
        adapter_name = self._ensure_lora(pipe, lora_path=lora_path, lora_weight_name=lora_weight_name)
        signature = inspect.signature(pipe.__call__).parameters
        kwargs = {
            "prompt": prompt,
            "image": self._prepare_image(image, width, height),
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
        }
        if negative_prompt and "negative_prompt" in signature:
            kwargs["negative_prompt"] = negative_prompt
        elif "negative_prompt" in signature:
            kwargs["negative_prompt"] = " "
        if "guidance_scale" in signature and guidance_scale != 1.0:
            kwargs["guidance_scale"] = guidance_scale
        if "true_cfg_scale" in signature:
            kwargs["true_cfg_scale"] = self.settings.qwen_edit_true_cfg_scale
        if strength is not None and "strength" in signature:
            kwargs["strength"] = strength
        if seed is not None:
            kwargs["generator"] = torch.Generator(device=self._generator_device()).manual_seed(seed)
        if adapter_name and "cross_attention_kwargs" in signature:
            kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        with torch.no_grad():
            result = pipe(**kwargs)
        return result.images[0].convert("RGB")

    def _ensure_lora(self, pipe, *, lora_path: Optional[str], lora_weight_name: Optional[str]) -> Optional[str]:
        if not lora_path:
            if hasattr(pipe, "disable_lora"):
                pipe.disable_lora()
            self._active_adapter = None
            return None
        adapter_name = "qwen_unblur_upscale"
        loaded_adapters = getattr(pipe, "get_list_adapters", lambda: {})()
        adapter_loaded = adapter_name in loaded_adapters or any(adapter_name in names for names in loaded_adapters.values()) if isinstance(loaded_adapters, dict) else False
        if not adapter_loaded:
            kwargs = {"adapter_name": adapter_name}
            if lora_weight_name:
                kwargs["weight_name"] = lora_weight_name
            pipe.load_lora_weights(lora_path, **kwargs)
        if self._active_adapter == adapter_name:
            return adapter_name
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters([adapter_name])
        elif hasattr(pipe, "enable_lora"):
            pipe.enable_lora()
        self._active_adapter = adapter_name
        return adapter_name

    def _prepare_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        if self.settings.qwen_edit_input_fit_mode == "cover":
            return ImageOps.fit(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        color = self.settings.qwen_edit_background_color
        return ImageOps.pad(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS, color=color, centering=(0.5, 0.5))

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
import asyncio
import gc
import inspect
import logging
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from app.config import Settings
from app.pipeline_utils import apply_pipeline_cpu_offload, apply_pipeline_memory_settings, get_pipeline_device_map_kwargs, uses_pipeline_device_map
from app.qwen_image_service import ModelManager, _model_manager


logger = logging.getLogger(__name__)


class QwenImageEditService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self._device = None
        self._dtype = None
        self._active_adapter = None
        self._model_name = "qwen_image_edit_2511"
        self._model_manager = _model_manager
        self._cpu_offload_enabled = False


    async def edit(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        image: Image.Image | list[Image.Image],
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

    async def prepare(self, *, lora_path: Optional[str] = None, lora_weight_name: Optional[str] = None) -> None:
        await asyncio.to_thread(self._prepare_sync, lora_path=lora_path, lora_weight_name=lora_weight_name)

    def _prepare_sync(self, *, lora_path: Optional[str], lora_weight_name: Optional[str]) -> None:
        pipe = self._get_pipeline()
        self._ensure_lora(pipe, lora_path=lora_path, lora_weight_name=lora_weight_name)

    def _edit_sync(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        image: Image.Image | list[Image.Image],
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
            "image": self._prepare_image_input(image, width, height),
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
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
            return adapter_name
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
        elif hasattr(pipe, "enable_lora"):
            pipe.enable_lora()
        self._active_adapter = adapter_name
        return adapter_name

    def _prepare_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        if self.settings.qwen_edit_input_fit_mode == "cover":
            return ImageOps.fit(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        color = self.settings.qwen_edit_background_color
        return ImageOps.pad(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS, color=color, centering=(0.5, 0.5))

    def _prepare_image_input(self, image: Image.Image | list[Image.Image], width: int, height: int) -> Image.Image | list[Image.Image]:
        if isinstance(image, list):
            return [self._prepare_image(item, width, height) for item in image]
        return self._prepare_image(image, width, height)

    def _get_pipeline(self):
        import torch
        
        # 检查是否需要切换模型
        current_model = self._get_model_name()
        if self._pipe is not None and self._model_name != current_model:
            logger.info(f"Model switch detected: {self._model_name} -> {current_model}")
            self._unload_pipeline()
        
        if self._pipe is not None:
            return self._pipe

        self._model_manager.unload_except(current_model)

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
            if self.settings.qwen_edit_device_map != "none":
                load_kwargs["device_map"] = self.settings.qwen_edit_device_map
        elif self.settings.qwen_edit_multi_gpu_enabled:
            load_kwargs.update(get_pipeline_device_map_kwargs(self.settings, torch, self._device))
        else:
            if self.settings.qwen_edit_device_map != "none":
                load_kwargs["device_map"] = self.settings.qwen_edit_device_map
            logger.info(
                "Qwen Edit multi-GPU loading disabled; loading with %s on device_map=%s device=%s",
                self.settings.qwen_edit_pipeline_class,
                load_kwargs.get("device_map"),
                self._device,
            )

        pipe = pipeline_cls.from_pretrained(self.settings.qwen_edit_model_path, **load_kwargs)
        cpu_offload_enabled = False
        device_map_enabled = uses_pipeline_device_map(load_kwargs)
        if device_map_enabled:
            pass
        elif apply_pipeline_cpu_offload(pipe, self.settings, self._device):
            cpu_offload_enabled = True
        else:
            pipe.to(self._device)
        apply_pipeline_memory_settings(pipe, self.settings)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self._log_device_map(pipe)

        self._pipe = pipe
        self._model_name = current_model
        self._cpu_offload_enabled = cpu_offload_enabled
        
        # 注册到模型管理器
        self._model_manager.register_model(self._model_name, pipe, self._estimate_model_size(torch), cpu_offload=cpu_offload_enabled or device_map_enabled)
        self._model_manager.activate_model(self._model_name)
        
        return pipe

    def _quantization_config(self, diffusers):
        if self.settings.qwen_edit_quantization == "none" or self._is_prequantized_model_path():
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

    def _is_prequantized_model_path(self) -> bool:
        model_path = Path(self.settings.qwen_edit_model_path)
        return model_path.is_dir() and (model_path / "quantization_info.json").exists()

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

    def _get_model_name(self) -> str:
        """获取当前模型名称"""
        model_path = self.settings.qwen_edit_model_path
        if "qwen-image-edit-2511" in model_path.lower():
            return "qwen_image_edit_2511"
        elif "qwen-image-2512" in model_path.lower():
            return "qwen_image_2512"
        else:
            return f"custom_edit_{hash(model_path) % 1000}"
    
    def _estimate_model_size(self, torch) -> float:
        """估算模型大小 (MB)"""
        # 4bit量化模型约3-4GB, FP16约14-16GB
        if self.settings.torch_dtype == "bfloat16" or self.settings.torch_dtype == "float16":
            return 15000  # ~15GB
        else:
            return 4000  # ~4GB (4bit量化)
    
    def _unload_pipeline(self) -> None:
        """卸载当前pipeline"""
        if self._pipe is not None:
            try:
                if not self._cpu_offload_enabled and not self._has_device_map() and hasattr(self._pipe, 'to'):
                    self._pipe.to('cpu')
                    logger.info(f"Moved pipeline to CPU before release: {self._model_name}")
                else:
                    logger.info(f"Releasing pipeline: {self._model_name}")
            except Exception as e:
                logger.warning(f"Failed to unload pipeline: {e}")
            self._pipe = None
            self._active_adapter = None
            self._cpu_offload_enabled = False
            self._model_manager.unregister_model(self._model_name)
            self._model_manager._release_torch_memory()

    def unload(self) -> None:
        """主动释放当前服务持有的 pipeline。"""
        self._unload_pipeline()

    def _has_device_map(self) -> bool:
        return bool(getattr(self._pipe, "hf_device_map", None))

    def _log_device_map(self, pipe) -> None:
        pipeline_device_map = getattr(pipe, "hf_device_map", None)
        if pipeline_device_map:
            logger.info("Qwen Edit pipeline device map: %s", pipeline_device_map)
        components = getattr(pipe, "components", {}) or {}
        for name, component in components.items():
            component_device_map = getattr(component, "hf_device_map", None)
            if component_device_map:
                logger.info("Qwen Edit component '%s' device map: %s", name, component_device_map)
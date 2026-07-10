import asyncio
import inspect
import logging
from typing import Any, Optional

from PIL import Image, ImageOps

from app.config import Settings
from app.pipeline_utils import apply_pipeline_cpu_offload, apply_pipeline_memory_settings
from app.qwen_image_service import ModelManager, _model_manager


logger = logging.getLogger(__name__)


class QwenImageEditService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self._device = None
        self._dtype = None
        self._model_name = "qwen_image_edit_2511"
        self._model_manager = _model_manager
        self._cpu_offload_enabled = False
        self._edit_lora_active = False
        self._lora_path: Optional[str] = None
        self._lora_weight_name: Optional[str] = None
        self._lora_scale: float = 1.0


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
        await asyncio.to_thread(
            self._prepare_sync,
            lora_path=lora_path,
            lora_weight_name=lora_weight_name,
            lora_scale=self.settings.qwen_unblur_upscale_lora_scale,
        )

    def _prepare_sync(self, *, lora_path: Optional[str], lora_weight_name: Optional[str], lora_scale: float) -> None:
        self._lora_path = lora_path
        self._lora_weight_name = lora_weight_name
        self._lora_scale = lora_scale
        self._get_pipeline()

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

        self._lora_path = lora_path
        self._lora_weight_name = lora_weight_name
        self._lora_scale = lora_scale

        pipe = self._get_pipeline()
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

        with torch.no_grad():
            result = pipe(**kwargs)
        return result.images[0].convert("RGB")

    def _apply_edit_lora(self, pipe) -> None:
        """按官方教程：pipe.load_lora_weights(path) — 无 adapter_name，无 set_adapters。"""
        if self._edit_lora_active:
            return
        if not self._lora_path:
            return
        if not hasattr(pipe, "load_lora_weights"):
            return

        logger.info("Loading Qwen Edit LoRA: path=%s weight=%s", self._lora_path, self._lora_weight_name)
        load_kwargs = {}
        if self._lora_weight_name:
            load_kwargs["weight_name"] = self._lora_weight_name
        pipe.load_lora_weights(self._lora_path, **load_kwargs)
        self._edit_lora_active = True
        logger.info("Qwen Edit LoRA loaded")

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
        if self._device.startswith("cuda"):
            load_kwargs["device_map"] = "cuda"
        if self.settings.hf_token and not self.settings.hf_token.startswith("replace-with"):
            load_kwargs["token"] = self.settings.hf_token

        pipe = pipeline_cls.from_pretrained(self.settings.qwen_edit_model_path, **load_kwargs)
        apply_pipeline_memory_settings(pipe, self.settings)
        self._apply_edit_lora(pipe)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        cpu_offload_enabled = apply_pipeline_cpu_offload(pipe, self.settings, self._device)
        self._pipe = pipe
        self._model_name = current_model
        self._cpu_offload_enabled = cpu_offload_enabled
        
        self._model_manager.register_model(self._model_name, pipe, self._estimate_model_size(torch), cpu_offload=cpu_offload_enabled)
        self._model_manager.activate_model(self._model_name)
        
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
                if not self._cpu_offload_enabled and hasattr(self._pipe, 'to'):
                    self._pipe.to('cpu')
                    logger.info(f"Moved pipeline to CPU before release: {self._model_name}")
                else:
                    logger.info(f"Releasing pipeline: {self._model_name}")
            except Exception as e:
                logger.warning(f"Failed to unload pipeline: {e}")
            self._pipe = None
            self._edit_lora_active = False
            self._cpu_offload_enabled = False
            self._model_manager.unregister_model(self._model_name)
            self._model_manager._release_torch_memory()

    def unload(self) -> None:
        """主动释放当前服务持有的 pipeline。"""
        self._unload_pipeline()
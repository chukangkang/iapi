import asyncio
import gc
import inspect
import logging
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageOps

from app.config import Settings
from app.pipeline_utils import apply_pipeline_cpu_offload, apply_pipeline_memory_settings, get_pipeline_device_map_kwargs, get_remaining_cuda_max_memory, uses_pipeline_device_map
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
        if self._edit_lora_active and "cross_attention_kwargs" in signature:
            kwargs["cross_attention_kwargs"] = {"scale": self._lora_scale}

        with torch.no_grad():
            result = pipe(**kwargs)
        return result.images[0].convert("RGB")

    def _apply_edit_lora(self, pipe) -> None:
        """与 QwenImage 2512 一致：fuse_lora + unload_lora_weights，消除 adapter 模式。"""
        if self._edit_lora_active:
            return
        if not self._lora_path:
            return
        if not hasattr(pipe, "load_lora_weights"):
            return

        logger.info(
            "Loading Qwen Edit LoRA: path=%s weight=%s scale=%s",
            self._lora_path,
            self._lora_weight_name,
            self._lora_scale,
        )
        adapter_name = "qwen_unblur_upscale"
        lora_kwargs = {"adapter_name": adapter_name}
        if self._lora_weight_name:
            lora_kwargs["weight_name"] = self._lora_weight_name
        pipe.load_lora_weights(self._lora_path, **lora_kwargs)
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters([adapter_name], adapter_weights=[self._lora_scale])
        if hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora(adapter_names=[adapter_name], lora_scale=self._lora_scale)
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
        self._edit_lora_active = True
        logger.info("Qwen Edit LoRA fused into model weights (scale=%.2f)", self._lora_scale)

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

        load_kwargs.update(get_pipeline_device_map_kwargs(self.settings, torch, self._device))
        device_map_enabled = uses_pipeline_device_map(load_kwargs)
        transformer_sharded = False
        transformer = None
        if device_map_enabled:
            transformer = self._load_sharded_transformer(diffusers, torch, load_kwargs)
            transformer_sharded = transformer is not None

        model_path = self.settings.qwen_edit_model_path
        pipeline_load_kwargs = load_kwargs.copy()
        if transformer_sharded:
            pipeline_load_kwargs.pop("device_map", None)
            pipeline_load_kwargs["transformer"] = transformer
            remaining_max_memory = get_remaining_cuda_max_memory(self.settings, torch, reserve_gib=2)
            if remaining_max_memory:
                pipeline_load_kwargs["max_memory"] = remaining_max_memory
                logger.info("Adjusted Qwen Edit pipeline max_memory after transformer load: %s", remaining_max_memory)

        pipe = pipeline_cls.from_pretrained(model_path, **pipeline_load_kwargs)

        device_map_enabled = device_map_enabled or transformer_sharded
        cpu_offload_enabled = False if device_map_enabled else apply_pipeline_cpu_offload(pipe, self.settings, self._device)
        if not cpu_offload_enabled and not device_map_enabled:
            pipe.to(self._device)
        apply_pipeline_memory_settings(pipe, self.settings)
        self._apply_edit_lora(pipe)
        if device_map_enabled:
            self._move_unsharded_components_to_device(pipe, torch)
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

    def _load_sharded_transformer(self, diffusers: Any, torch: Any, load_kwargs: dict[str, Any]) -> Optional[Any]:
        transformer_cls = getattr(diffusers, "QwenImageTransformer2DModel", None)
        if transformer_cls is None:
            logger.warning("QwenImageTransformer2DModel is unavailable; falling back to pipeline-level device_map")
            return None
        if self._looks_like_single_file(self.settings.qwen_edit_model_path):
            logger.warning("Single-file Qwen Edit model cannot load transformer subfolder separately; falling back to pipeline-level device_map")
            return None

        transformer_kwargs = load_kwargs.copy()
        transformer_kwargs["device_map"] = transformer_kwargs.get("device_map", "balanced")
        logger.info(
            "Loading Qwen Edit transformer with layer-level multi-GPU sharding: device_map=%s max_memory=%s",
            transformer_kwargs.get("device_map"),
            transformer_kwargs.get("max_memory"),
        )
        transformer = transformer_cls.from_pretrained(
            self.settings.qwen_edit_model_path,
            subfolder="transformer",
            **transformer_kwargs,
        )
        device_map = getattr(transformer, "hf_device_map", None)
        if device_map:
            logger.info("Qwen Edit transformer device map: %s", device_map)
        return transformer

    def _looks_like_single_file(self, model_path: str) -> bool:
        suffix = Path(model_path).suffix.lower()
        return suffix in {".safetensors", ".ckpt", ".pt", ".pth"}


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
            self._edit_lora_active = False
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
        transformer = getattr(pipe, "transformer", None)
        transformer_device_map = getattr(transformer, "hf_device_map", None)
        if transformer_device_map:
            per_device: dict[str, int] = {}
            for device in transformer_device_map.values():
                key = str(device)
                per_device[key] = per_device.get(key, 0) + 1
            summary = ", ".join(f"{device}={count}" for device, count in sorted(per_device.items(), key=lambda item: item[0]))
            logger.info("Qwen Edit transformer device map: %s", summary)

    def _move_unsharded_components_to_device(self, pipe, torch) -> None:
        """将未被 device_map 管理的组件移到显存使用最少的 GPU。"""
        components = getattr(pipe, "components", {}) or {}
        for name, component in components.items():
            if not isinstance(component, torch.nn.Module):
                continue
            if getattr(component, "hf_device_map", None):
                continue
            target_device = self._least_used_cuda_device(torch)
            try:
                component.to(target_device)
                logger.info("Moved Qwen Edit component '%s' to %s", name, target_device)
            except Exception as exc:
                logger.debug("Failed to move Qwen Edit component '%s' to %s: %s", name, target_device, exc)

    def _least_used_cuda_device(self, torch) -> str:
        if not self._device or not self._device.startswith("cuda") or not torch.cuda.is_available():
            return self._device
        device_count = min(self.settings.model_gpu_count, torch.cuda.device_count(), 4)
        if device_count <= 1:
            return self._device
        memory_by_device: list[tuple[int, int]] = []
        for index in range(device_count):
            try:
                memory_by_device.append((torch.cuda.memory_allocated(index), index))
            except Exception as exc:
                logger.debug("Failed to read allocated memory for cuda:%s: %s", index, exc)
        if not memory_by_device:
            return self._device
        _, device_index = min(memory_by_device)
        return f"cuda:{device_index}"
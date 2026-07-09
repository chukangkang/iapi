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


logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器,支持模型在CPU/GPU间动态切换"""
    
    def __init__(self):
        self._models: dict[str, any] = {}
        self._active_model: Optional[str] = None
        self._model_sizes: dict[str, float] = {}  # MB
        self._last_access: dict[str, float] = {}
        self._cpu_offload_models: set[str] = set()
        
    def register_model(self, name: str, pipe: any, size_mb: float, *, cpu_offload: bool = False) -> None:
        """注册模型"""
        self._models[name] = pipe
        self._model_sizes[name] = size_mb
        self._last_access[name] = time.time()
        if cpu_offload:
            self._cpu_offload_models.add(name)
        else:
            self._cpu_offload_models.discard(name)
        logger.info(f"Registered model '{name}': {size_mb:.1f} MB")
    
    def unregister_model(self, name: str) -> None:
        """注销模型"""
        if name in self._models:
            pipe = self._models.pop(name)
            try:
                if name not in self._cpu_offload_models and hasattr(pipe, 'to'):
                    pipe.to('cpu')
                    logger.info(f"Moved model '{name}' to CPU before release")
                else:
                    logger.info(f"Releasing model '{name}'")
            except Exception as e:
                logger.warning(f"Failed to unload model '{name}': {e}")
            del pipe
            self._model_sizes.pop(name, None)
            self._last_access.pop(name, None)
            self._cpu_offload_models.discard(name)
            if self._active_model == name:
                self._active_model = None
            self._release_torch_memory()

    def unload_except(self, keep_name: str) -> None:
        """释放除 keep_name 以外的已注册模型引用。"""
        for name in list(self._models.keys()):
            if name != keep_name:
                self.unregister_model(name)
    
    def activate_model(self, name: str) -> bool:
        """激活模型,如果需要则从CPU加载到GPU"""
        if name not in self._models:
            return False
        
        if self._active_model == name:
            return True
        
        import torch
        
        # 卸载当前模型
        if self._active_model and self._active_model != name:
            self._unload_active_model()
        
        # 加载到GPU
        pipe = self._models[name]
        try:
            if name not in self._cpu_offload_models and hasattr(pipe, 'to'):
                pipe.to('cuda')
            self._active_model = name
            self._last_access[name] = time.time()
            if name in self._cpu_offload_models:
                logger.info(f"Activated model '{name}' with CPU offload")
            else:
                logger.info(f"Activated model '{name}' on GPU")
            return True
        except Exception as e:
            logger.error(f"Failed to activate model '{name}': {e}")
            return False
    
    def _unload_active_model(self) -> None:
        """卸载当前活跃模型"""
        if self._active_model:
            pipe = self._models.get(self._active_model)
            if pipe and self._active_model not in self._cpu_offload_models and hasattr(pipe, 'to'):
                try:
                    pipe.to('cpu')
                    logger.info(f"Unloaded model '{self._active_model}' to CPU")
                except Exception as e:
                    logger.warning(f"Failed to unload model '{self._active_model}': {e}")
            self._active_model = None
    
    def clear_all(self) -> None:
        """清除所有模型"""
        for name in list(self._models.keys()):
            self.unregister_model(name)
        self._active_model = None
        self._release_torch_memory()
        logger.info("Cleared all models")

    def _release_torch_memory(self) -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:
            logger.debug("Failed to release torch cache: %s", exc)
    
    def get_model(self, name: str) -> Optional[any]:
        """获取模型"""
        return self._models.get(name)
    
    def get_active_model(self) -> Optional[str]:
        """获取当前活跃模型名称"""
        return self._active_model
    
    def get_memory_info(self) -> dict:
        """获取内存信息"""
        import torch
        
        info = {
            "active_model": self._active_model,
            "registered_models": list(self._models.keys()),
            "model_sizes": self._model_sizes.copy()
        }
        
        if torch.cuda.is_available():
            info["cuda"] = {
                "total_mb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
                "allocated_mb": torch.cuda.memory_allocated(0) / (1024 ** 2),
                "reserved_mb": torch.cuda.memory_reserved(0) / (1024 ** 2),
                "free_mb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024 ** 2)
            }
        
        return info


# 全局模型管理器
_model_manager = ModelManager()


class QwenImageService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self._device = None
        self._dtype = None
        self._model_name = "qwen_image_2512"
        self._cpu_offload_enabled = False
        self._turbo_lora_active = False

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
    ) -> Image.Image:
        import torch

        pipe = self._get_pipeline()
        signature = inspect.signature(pipe.__call__).parameters
        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": self._effective_num_inference_steps(num_inference_steps),
        }
        if image is not None:
            image_argument = self._image_argument_name(signature)
            if image_argument is None:
                raise RuntimeError("DiffusionPipeline does not support image input for /v1/images/edits.")
            kwargs[image_argument] = self._prepare_image(image, width, height)
        if negative_prompt and "negative_prompt" in signature:
            kwargs["negative_prompt"] = negative_prompt
        elif "negative_prompt" in signature:
            kwargs["negative_prompt"] = " "
        if "true_cfg_scale" in signature:
            kwargs["true_cfg_scale"] = self._effective_true_cfg_scale()
        if seed is not None:
            kwargs["generator"] = torch.Generator(device=self._generator_device()).manual_seed(seed)

        logger.info(
            "Qwen Image generation: size=%sx%s steps=%s seed=%s true_cfg_scale=%s",
            width,
            height,
            num_inference_steps,
            seed,
            kwargs.get("true_cfg_scale"),
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
        import torch
        
        # 检查是否需要切换模型
        current_model = self._get_model_name()
        if self._pipe is not None and self._model_name != current_model:
            logger.info(f"Model switch detected: {self._model_name} -> {current_model}")
            self._unload_pipeline()
        
        if self._pipe is not None:
            return self._pipe

        _model_manager.unload_except(current_model)

        import diffusers

        pipeline_cls = diffusers.DiffusionPipeline
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

        model_path = self.settings.qwen_image_model_path
        pipeline_load_kwargs = load_kwargs.copy()
        if transformer_sharded:
            pipeline_load_kwargs["transformer"] = transformer
            remaining_max_memory = get_remaining_cuda_max_memory(self.settings, torch, reserve_gib=2)
            if remaining_max_memory:
                pipeline_load_kwargs["max_memory"] = remaining_max_memory
                logger.info("Adjusted Qwen Image pipeline max_memory after transformer load: %s", remaining_max_memory)
        if self._looks_like_single_file(model_path) and hasattr(pipeline_cls, "from_single_file"):
            pipe = pipeline_cls.from_single_file(model_path, **pipeline_load_kwargs)
        else:
            pipe = pipeline_cls.from_pretrained(model_path, **pipeline_load_kwargs)

        device_map_enabled = device_map_enabled or transformer_sharded
        cpu_offload_enabled = False if device_map_enabled else apply_pipeline_cpu_offload(pipe, self.settings, self._device)
        if not cpu_offload_enabled:
            if not device_map_enabled:
                pipe.to(self._device)
        apply_pipeline_memory_settings(pipe, self.settings)
        self._apply_turbo_lora(pipe)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self._log_device_map(pipe)

        self._pipe = pipe
        self._model_name = current_model
        self._cpu_offload_enabled = cpu_offload_enabled
        
        # 注册到模型管理器
        _model_manager.register_model(self._model_name, pipe, self._estimate_model_size(torch), cpu_offload=cpu_offload_enabled or device_map_enabled)
        _model_manager.activate_model(self._model_name)
        
        return pipe

    def _effective_num_inference_steps(self, requested_steps: int) -> int:
        if self.settings.qwen_image_turbo_lora_enabled:
            return self.settings.qwen_image_turbo_steps
        return requested_steps

    def _effective_true_cfg_scale(self) -> float:
        if self.settings.qwen_image_turbo_lora_enabled:
            return self.settings.qwen_image_turbo_true_cfg_scale
        return self.settings.qwen_image_true_cfg_scale

    def _apply_turbo_lora(self, pipe: Any) -> None:
        if not self.settings.qwen_image_turbo_lora_enabled or self._turbo_lora_active:
            return
        if not hasattr(pipe, "load_lora_weights"):
            raise RuntimeError("Current DiffusionPipeline does not support load_lora_weights for Qwen Image Turbo LoRA.")

        logger.info(
            "Loading Qwen Image Turbo LoRA: path=%s weight=%s scale=%s fuse=%s",
            self.settings.qwen_image_turbo_lora_path,
            self.settings.qwen_image_turbo_lora_weight_name,
            self.settings.qwen_image_turbo_lora_scale,
            self.settings.qwen_image_turbo_lora_fuse,
        )
        pipe.load_lora_weights(
            self.settings.qwen_image_turbo_lora_path,
            weight_name=self.settings.qwen_image_turbo_lora_weight_name,
            adapter_name="qwen_image_turbo_2steps",
        )
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(["qwen_image_turbo_2steps"], adapter_weights=[self.settings.qwen_image_turbo_lora_scale])
        if self.settings.qwen_image_turbo_lora_fuse and hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora(adapter_names=["qwen_image_turbo_2steps"], lora_scale=self.settings.qwen_image_turbo_lora_scale)
            if hasattr(pipe, "unload_lora_weights"):
                pipe.unload_lora_weights()
        self._apply_turbo_scheduler_config(pipe)
        self._turbo_lora_active = True

    def _apply_turbo_scheduler_config(self, pipe: Any) -> None:
        scheduler = getattr(pipe, "scheduler", None)
        if scheduler is None:
            logger.warning("Qwen Image Turbo LoRA scheduler config skipped: pipeline has no scheduler")
            return
        scheduler_kwargs = {
            "exponential_shift_mu": self.settings.qwen_image_turbo_scheduler_exponential_shift_mu,
            "use_dynamic_shifting": self.settings.qwen_image_turbo_scheduler_use_dynamic_shifting,
            "shift_terminal": self.settings.qwen_image_turbo_scheduler_shift_terminal,
        }
        try:
            pipe.scheduler = scheduler.__class__.from_config(scheduler.config, **scheduler_kwargs)
            logger.info("Applied Qwen Image Turbo scheduler config: %s", scheduler_kwargs)
        except Exception as exc:
            logger.warning("Failed to apply Qwen Image Turbo scheduler config: %s", exc)

    def _load_sharded_transformer(self, diffusers: Any, torch: Any, load_kwargs: dict[str, Any]) -> Optional[Any]:
        transformer_cls = getattr(diffusers, "QwenImageTransformer2DModel", None)
        if transformer_cls is None:
            logger.warning("QwenImageTransformer2DModel is unavailable; falling back to pipeline-level device_map")
            return None
        if self._looks_like_single_file(self.settings.qwen_image_model_path):
            logger.warning("Single-file Qwen Image model cannot load transformer subfolder separately; falling back to pipeline-level device_map")
            return None

        transformer_kwargs = load_kwargs.copy()
        transformer_kwargs["device_map"] = transformer_kwargs.get("device_map", "balanced")
        logger.info(
            "Loading Qwen Image transformer with layer-level multi-GPU sharding: device_map=%s max_memory=%s",
            transformer_kwargs.get("device_map"),
            transformer_kwargs.get("max_memory"),
        )
        transformer = transformer_cls.from_pretrained(
            self.settings.qwen_image_model_path,
            subfolder="transformer",
            **transformer_kwargs,
        )
        device_map = getattr(transformer, "hf_device_map", None)
        if device_map:
            logger.info("Qwen Image transformer device map: %s", device_map)
        return transformer

    def _move_unsharded_components_to_device(self, pipe: Any, torch: Any) -> None:
        components = getattr(pipe, "components", {}) or {}
        for name, component in components.items():
            if not isinstance(component, torch.nn.Module):
                continue
            if getattr(component, "hf_device_map", None):
                continue
            target_device = self._least_used_cuda_device(torch)
            try:
                component.to(target_device)
                logger.info("Moved Qwen Image component '%s' to %s", name, target_device)
            except Exception as exc:
                logger.debug("Failed to move Qwen Image component '%s' to %s: %s", name, target_device, exc)

    def _least_used_cuda_device(self, torch: Any) -> str:
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

    def _log_device_map(self, pipe: Any) -> None:
        pipeline_device_map = getattr(pipe, "hf_device_map", None)
        if pipeline_device_map:
            logger.info("Qwen Image pipeline device map: %s", pipeline_device_map)
        transformer = getattr(pipe, "transformer", None)
        transformer_device_map = getattr(transformer, "hf_device_map", None)
        if transformer_device_map:
            logger.info("Qwen Image transformer device map: %s", transformer_device_map)
    
    def _get_model_name(self) -> str:
        """获取当前模型名称"""
        model_path = self.settings.qwen_image_model_path
        if "qwen-image-2512" in model_path.lower():
            return "qwen_image_2512"
        elif "qwen-image-edit" in model_path.lower():
            return "qwen_image_edit_2511"
        else:
            return f"custom_{hash(model_path) % 1000}"
    
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
            self._cpu_offload_enabled = False
            self._turbo_lora_active = False
            _model_manager.unregister_model(self._model_name)
            _model_manager._release_torch_memory()

    def unload(self) -> None:
        """主动释放当前服务持有的 pipeline。"""
        self._unload_pipeline()

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
        if self.settings.torch_dtype == "bfloat16" and self._device == "cpu":
            return torch.float32
        if self.settings.torch_dtype == "auto":
            return torch.bfloat16 if self._device == "cuda" else torch.float32
        return getattr(torch, self.settings.torch_dtype)

    def _has_device_map(self) -> bool:
        return bool(getattr(self._pipe, "hf_device_map", None))

    def _generator_device(self) -> str:
        if not self._device or self._device == "mps":
            return "cpu"
        return self._device

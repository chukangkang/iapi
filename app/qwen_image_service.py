import asyncio
import gc
import inspect
import logging
import re
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from app.config import Settings
from app.pipeline_utils import apply_pipeline_cpu_offload, apply_pipeline_memory_settings


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
        self._lora_loaded = False
        self._model_name = "qwen_image_2512"
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

        cpu_offload_enabled = apply_pipeline_cpu_offload(pipe, self.settings, self._device)
        if not cpu_offload_enabled:
            pipe.to(self._device)
        apply_pipeline_memory_settings(pipe, self.settings)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        self._load_lora(pipe)
        self._apply_scheduler_config(pipe)
        self._pipe = pipe
        self._model_name = current_model
        self._cpu_offload_enabled = cpu_offload_enabled
        
        # 注册到模型管理器
        _model_manager.register_model(self._model_name, pipe, self._estimate_model_size(torch), cpu_offload=cpu_offload_enabled)
        _model_manager.activate_model(self._model_name)
        
        return pipe
    
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
                if not self._cpu_offload_enabled and hasattr(self._pipe, 'to'):
                    self._pipe.to('cpu')
                    logger.info(f"Moved pipeline to CPU before release: {self._model_name}")
                else:
                    logger.info(f"Releasing pipeline: {self._model_name}")
            except Exception as e:
                logger.warning(f"Failed to unload pipeline: {e}")
            self._pipe = None
            self._lora_loaded = False
            self._cpu_offload_enabled = False
            _model_manager.unregister_model(self._model_name)
            _model_manager._release_torch_memory()

    def unload(self) -> None:
        """主动释放当前服务持有的 pipeline。"""
        self._unload_pipeline()

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

import asyncio
import inspect
import logging
import math
import threading
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
        self._pipe_lock = threading.Lock()


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

        with self._pipe_lock, torch.no_grad():
            result = pipe(**kwargs)
        return result.images[0].convert("RGB")

    def _apply_edit_lora(self, pipe) -> None:
        if self._edit_lora_active:
            return
        if not hasattr(pipe, "load_lora_weights"):
            return

        adapter_names = []
        adapter_weights = []
        if self.settings.qwen_edit_lightning_lora_enabled:
            logger.info(
                "Loading Qwen Edit Lightning LoRA: path=%s weight=%s scale=%s",
                self.settings.qwen_edit_lightning_lora_path,
                self.settings.qwen_edit_lightning_lora_weight_name,
                self.settings.qwen_edit_lightning_lora_scale,
            )
            pipe.load_lora_weights(
                self.settings.qwen_edit_lightning_lora_path,
                weight_name=self.settings.qwen_edit_lightning_lora_weight_name,
                adapter_name="qwen_edit_lightning",
            )
            adapter_names.append("qwen_edit_lightning")
            adapter_weights.append(self.settings.qwen_edit_lightning_lora_scale)
        if self._lora_path:
            logger.info("Loading Qwen Edit Unblur LoRA: path=%s weight=%s scale=%s", self._lora_path, self._lora_weight_name, self._lora_scale)
            load_kwargs = {"adapter_name": "qwen_edit_unblur"}
            if self._lora_weight_name:
                load_kwargs["weight_name"] = self._lora_weight_name
            pipe.load_lora_weights(self._lora_path, **load_kwargs)
            adapter_names.append("qwen_edit_unblur")
            adapter_weights.append(self._lora_scale)
        if adapter_names and hasattr(pipe, "set_adapters"):
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        self._edit_lora_active = True
        logger.info("Qwen Edit LoRA adapters active: %s", adapter_names)

    def _apply_edit_scheduler(self, pipe) -> None:
        scheduler = getattr(pipe, "scheduler", None)
        if scheduler is None:
            logger.warning("Qwen Edit scheduler config skipped: pipeline has no scheduler")
            return
        base_shift = self.settings.qwen_edit_scheduler_base_shift
        scheduler_kwargs = {
            "base_image_seq_len": 256,
            "base_shift": base_shift,
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": base_shift,
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        pipe.scheduler = scheduler.__class__.from_config(scheduler.config, **scheduler_kwargs)
        logger.info("Applied Qwen Edit Lightning scheduler: base_shift=%.4f", base_shift)

    def _configure_edit_pipeline(self, pipe) -> None:
        if self.settings.qwen_edit_lightning_lora_enabled:
            self._apply_edit_scheduler(pipe)
        self._apply_edit_lora(pipe)

    def _prepare_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        image = image.convert("RGB")
        if self.settings.qwen_edit_input_fit_mode == "cover":
            return ImageOps.fit(
                image,
                (width, height),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        color = self.settings.qwen_edit_background_color
        return ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS, color=color, centering=(0.5, 0.5))

    def _prepare_image_input(self, image: Image.Image | list[Image.Image], width: int, height: int) -> Image.Image | list[Image.Image]:
        if isinstance(image, list):
            return [self._prepare_image(item, width, height) for item in image]
        return self._prepare_image(image, width, height)

    def align_to_reference(self, image: Image.Image, reference: Image.Image) -> Image.Image:
        if not self.settings.qwen_unblur_upscale_alignment_enabled:
            return image
        try:
            enhanced = image.convert("RGB")
            target = self._prepare_image(reference, enhanced.width, enhanced.height)
            if self.settings.qwen_unblur_upscale_alignment_mode == "dense":
                densely_aligned = self._align_dense_flow(enhanced, target)
                if densely_aligned is not None:
                    return densely_aligned
            if self.settings.qwen_unblur_upscale_alignment_mode == "similarity":
                similarity_aligned = self._align_similarity(enhanced, target)
                if similarity_aligned is not None:
                    return similarity_aligned
            max_shift = self.settings.qwen_unblur_upscale_alignment_max_shift
            shift_x, shift_y, alignment_error = self._estimate_translation(target, enhanced, max_shift)
            if abs(shift_x) > max_shift or abs(shift_y) > max_shift:
                logger.warning(
                    "Skipping Qwen unblur alignment: estimated shift exceeds limit shift=(%.2f, %.2f) limit=%s",
                    shift_x,
                    shift_y,
                    max_shift,
                )
                return enhanced
            aligned = target.copy()
            source_left = max(0, -shift_x)
            source_top = max(0, -shift_y)
            destination_left = max(0, shift_x)
            destination_top = max(0, shift_y)
            copy_width = min(enhanced.width - source_left, enhanced.width - destination_left)
            copy_height = min(enhanced.height - source_top, enhanced.height - destination_top)
            if copy_width <= 0 or copy_height <= 0:
                return enhanced
            aligned.paste(
                enhanced.crop((source_left, source_top, source_left + copy_width, source_top + copy_height)),
                (destination_left, destination_top),
            )
            logger.info(
                "Aligned Qwen unblur output to source: shift=(%s, %s) alignment_error=%.4f",
                shift_x,
                shift_y,
                alignment_error,
            )
            return aligned
        except Exception as exc:
            logger.warning("Qwen unblur output alignment failed; using unaligned image: %s", exc)
            return image

    def _align_similarity(self, enhanced: Image.Image, target: Image.Image) -> Optional[Image.Image]:
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.warning("Similarity alignment requires opencv-python and numpy; falling back to translation")
            return None

        max_side = self.settings.qwen_unblur_upscale_alignment_max_side
        sample_scale = min(1.0, max_side / max(enhanced.width, enhanced.height))
        sample_size = (max(64, round(enhanced.width * sample_scale)), max(64, round(enhanced.height * sample_scale)))
        enhanced_gray = np.asarray(enhanced.resize(sample_size, Image.Resampling.BILINEAR).convert("L"))
        target_gray = np.asarray(target.resize(sample_size, Image.Resampling.BILINEAR).convert("L"))
        detector = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
        enhanced_points, enhanced_descriptors = detector.detectAndCompute(enhanced_gray, None)
        target_points, target_descriptors = detector.detectAndCompute(target_gray, None)
        if enhanced_descriptors is None or target_descriptors is None:
            logger.warning("Similarity alignment found insufficient image features; falling back to translation")
            return None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(enhanced_descriptors, target_descriptors, k=2)
        good_matches = [first for first, second in matches if first.distance < 0.75 * second.distance]
        if len(good_matches) < 8:
            logger.warning("Similarity alignment found only %s reliable matches; falling back to translation", len(good_matches))
            return None
        source_points = np.float32([enhanced_points[match.queryIdx].pt for match in good_matches])
        target_points_array = np.float32([target_points[match.trainIdx].pt for match in good_matches])
        matrix, inliers = cv2.estimateAffinePartial2D(
            source_points,
            target_points_array,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=3000,
            confidence=0.99,
            refineIters=20,
        )
        if matrix is None or inliers is None or int(inliers.sum()) < 6:
            logger.warning("Similarity alignment could not estimate a reliable transform; falling back to translation")
            return None
        scale = float((matrix[0, 0] ** 2 + matrix[0, 1] ** 2) ** 0.5)
        rotation_degrees = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
        shift_x = float(matrix[0, 2] / sample_scale)
        shift_y = float(matrix[1, 2] / sample_scale)
        if not self._similarity_transform_is_safe(scale, rotation_degrees, shift_x, shift_y):
            logger.warning(
                "Rejected unsafe Qwen similarity alignment: scale=%.4f rotation=%.2f shift=(%.2f, %.2f)",
                scale,
                rotation_degrees,
                shift_x,
                shift_y,
            )
            return None
        full_matrix = matrix.copy()
        full_matrix[0, 2] /= sample_scale
        full_matrix[1, 2] /= sample_scale
        aligned = cv2.warpAffine(
            np.asarray(enhanced),
            full_matrix,
            (enhanced.width, enhanced.height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        logger.info(
            "Similarity-aligned Qwen unblur output: scale=%.4f rotation=%.2f shift=(%.2f, %.2f) inliers=%s/%s",
            scale,
            rotation_degrees,
            shift_x,
            shift_y,
            int(inliers.sum()),
            len(good_matches),
        )
        return Image.fromarray(aligned).convert("RGB")

    def _similarity_transform_is_safe(
        self,
        scale: float,
        rotation_degrees: float,
        shift_x: float,
        shift_y: float,
    ) -> bool:
        return (
            abs(scale - 1.0) <= self.settings.qwen_unblur_upscale_alignment_max_scale_delta
            and abs(rotation_degrees) <= self.settings.qwen_unblur_upscale_alignment_max_rotation_degrees
            and abs(shift_x) <= self.settings.qwen_unblur_upscale_alignment_max_shift
            and abs(shift_y) <= self.settings.qwen_unblur_upscale_alignment_max_shift
        )

    def _align_dense_flow(self, enhanced: Image.Image, target: Image.Image) -> Optional[Image.Image]:
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.warning("Dense Qwen unblur alignment requires opencv-python and numpy; falling back to translation")
            return None

        max_side = self.settings.qwen_unblur_upscale_alignment_max_side
        scale = min(1.0, max_side / max(enhanced.width, enhanced.height))
        sample_size = (max(32, round(enhanced.width * scale)), max(32, round(enhanced.height * scale)))
        target_sample = np.asarray(target.resize(sample_size, Image.Resampling.BILINEAR).convert("L"))
        enhanced_sample = np.asarray(enhanced.resize(sample_size, Image.Resampling.BILINEAR).convert("L"))
        flow_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow_estimator.setUseSpatialPropagation(True)
        flow = flow_estimator.calc(target_sample, enhanced_sample, None)
        if scale != 1.0:
            flow = cv2.resize(flow, (enhanced.width, enhanced.height), interpolation=cv2.INTER_LINEAR) / scale
        strength = self.settings.qwen_unblur_upscale_alignment_flow_strength
        flow *= strength
        max_shift = float(self.settings.qwen_unblur_upscale_alignment_max_shift)
        np.clip(flow, -max_shift, max_shift, out=flow)
        grid_x, grid_y = np.meshgrid(
            np.arange(enhanced.width, dtype=np.float32),
            np.arange(enhanced.height, dtype=np.float32),
        )
        remapped = cv2.remap(
            np.asarray(enhanced),
            grid_x + flow[..., 0],
            grid_y + flow[..., 1],
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        logger.info(
            "Dense-aligned Qwen unblur output to source: mean_flow=%.2f max_flow=%.2f strength=%.2f",
            float(magnitudes.mean()),
            float(magnitudes.max()),
            strength,
        )
        return Image.fromarray(remapped).convert("RGB")

    @staticmethod
    def _estimate_translation(target: Image.Image, enhanced: Image.Image, max_shift: int) -> tuple[int, int, float]:
        scale = min(1.0, 1024 / max(target.width, target.height))
        sample_size = (max(16, round(target.width * scale)), max(16, round(target.height * scale)))
        target_sample = target.convert("L").resize(sample_size, Image.Resampling.BILINEAR)
        enhanced_sample = enhanced.convert("L").resize(sample_size, Image.Resampling.BILINEAR)
        target_columns, target_rows = QwenImageEditService._edge_projections(target_sample)
        enhanced_columns, enhanced_rows = QwenImageEditService._edge_projections(enhanced_sample)
        sample_max_shift = max(1, round(max_shift * scale)) if max_shift else 0
        sample_shift_x, error_x = QwenImageEditService._best_projection_shift(
            target_columns, enhanced_columns, sample_max_shift
        )
        sample_shift_y, error_y = QwenImageEditService._best_projection_shift(
            target_rows, enhanced_rows, sample_max_shift
        )
        return round(sample_shift_x / scale), round(sample_shift_y / scale), (error_x + error_y) / 2

    @staticmethod
    def _edge_projections(image: Image.Image) -> tuple[list[float], list[float]]:
        pixels = image.load()
        columns = [0.0] * image.width
        rows = [0.0] * image.height
        for y in range(1, image.height):
            for x in range(1, image.width):
                value = abs(pixels[x, y] - pixels[x - 1, y]) + abs(pixels[x, y] - pixels[x, y - 1])
                columns[x] += value
                rows[y] += value
        return columns, rows

    @staticmethod
    def _best_projection_shift(target: list[float], enhanced: list[float], max_shift: int) -> tuple[int, float]:
        best_shift = 0
        best_error = float("inf")
        for shift in range(-max_shift, max_shift + 1):
            start = max(0, shift)
            end = min(len(target), len(enhanced) + shift)
            if end <= start:
                continue
            error = sum(abs(target[index] - enhanced[index - shift]) for index in range(start, end)) / (end - start)
            if error < best_error:
                best_shift = shift
                best_error = error
        return best_shift, best_error

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
        # LoRA 必须在 _move_unsharded_components_to_device 之前，
        # 因为 load_lora_weights 可能重新分配设备。之后再把 CPU 组件移 GPU。
        self._configure_edit_pipeline(pipe)
        if device_map_enabled:
            self._move_unsharded_components_to_device(pipe, torch)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        self._pipe = pipe
        self._model_name = current_model
        self._cpu_offload_enabled = cpu_offload_enabled

        self._model_manager.register_model(self._model_name, pipe, self._estimate_model_size(torch), cpu_offload=cpu_offload_enabled or device_map_enabled)
        self._model_manager.activate_model(self._model_name)

        return pipe

    def _load_sharded_transformer(self, diffusers, torch, load_kwargs):
        transformer_cls = getattr(diffusers, "QwenImageTransformer2DModel", None)
        if transformer_cls is None:
            logger.warning("QwenImageTransformer2DModel unavailable; falling back to pipeline-level device_map")
            return None
        if self._looks_like_single_file(self.settings.qwen_edit_model_path):
            logger.warning("Single-file model cannot load transformer subfolder; falling back to pipeline-level device_map")
            return None

        transformer_kwargs = load_kwargs.copy()
        transformer_kwargs["device_map"] = transformer_kwargs.get("device_map", "balanced")
        logger.info("Loading Qwen Edit transformer with layer-level sharding: device_map=%s", transformer_kwargs["device_map"])
        transformer = transformer_cls.from_pretrained(self.settings.qwen_edit_model_path, subfolder="transformer", **transformer_kwargs)
        device_map = getattr(transformer, "hf_device_map", None)
        if device_map:
            per_device: dict[str, int] = {}
            for d in device_map.values():
                key = str(d)
                per_device[key] = per_device.get(key, 0) + 1
            logger.info("Qwen Edit transformer device map: %s", ", ".join(f"{k}={v}" for k, v in sorted(per_device.items())))
        return transformer

    def _looks_like_single_file(self, model_path: str) -> bool:
        suffix = Path(model_path).suffix.lower()
        return suffix in {".safetensors", ".ckpt", ".pt", ".pth"}

    def _move_unsharded_components_to_device(self, pipe, torch) -> None:
        """将未被 device_map 管理的组件移到显存使用最少的 GPU。"""
        components = getattr(pipe, "components", {}) or {}
        for name, component in components.items():
            if not isinstance(component, torch.nn.Module):
                continue
            if getattr(component, "hf_device_map", None):
                continue
            try:
                current_device = str(next(component.parameters()).device)
            except StopIteration:
                continue
            if current_device != "cpu":
                continue
            target_device = self._least_used_cuda_device(torch)
            try:
                component.to(target_device)
                logger.info("Moved Qwen Edit component '%s' to %s", name, target_device)
            except Exception as exc:
                logger.warning("Failed to move Qwen Edit component '%s' to %s: %s", name, target_device, exc)

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
            except Exception:
                pass
        if not memory_by_device:
            return self._device
        _, device_index = min(memory_by_device)
        return f"cuda:{device_index}"

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
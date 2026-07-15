from __future__ import annotations

import importlib
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from app.config import Settings


logger = logging.getLogger(__name__)


def _comfyui_import_error(exc: BaseException) -> str:
    missing_module = getattr(exc, "name", None)
    if not missing_module and isinstance(exc, ModuleNotFoundError):
        message = str(exc)
        marker = "No module named "
        if marker in message:
            missing_module = message.split(marker, 1)[1].strip().strip("'\"")
    install_hint = ""
    if missing_module:
        install_hint = (
            f" Missing Python module: {missing_module!r}. Install Worker dependencies with "
            f"`{sys.executable} -m pip install -r requirements-worker.txt`; for this specific error, "
            f"`{sys.executable} -m pip install {missing_module}`."
        )
    return (
        "Failed to import the local ComfyUI core. Install COMFYUI_PATH/requirements.txt and "
        "requirements-worker.txt into the same Python environment used by this Worker, then verify "
        f"that checkout starts normally.{install_hint} Original error: {exc}"
    )


class ComfyUIRuntime:
    """Loads the official ComfyUI Python core in this worker process."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self._validate_root()
        root_string = str(self.root)
        if root_string in sys.path:
            sys.path.remove(root_string)
        sys.path.insert(0, root_string)

        try:
            self.torch = importlib.import_module("torch")
            self.folder_paths = importlib.import_module("folder_paths")
            self.nodes = importlib.import_module("nodes")
            self.sd = importlib.import_module("comfy.sd")
            self.utils = importlib.import_module("comfy.utils")
            self.model_management = importlib.import_module("comfy.model_management")
            advanced = importlib.import_module("comfy_extras.nodes_model_advanced")
            qwen = importlib.import_module("comfy_extras.nodes_qwen")
        except Exception as exc:
            raise RuntimeError(_comfyui_import_error(exc)) from exc
        self._sampling_node = advanced.ModelSamplingAuraFlow()
        self._qwen_encode_node = qwen.TextEncodeQwenImageEditPlus
        self._negative_encode_node = self.nodes.CLIPTextEncode()
        self._latent_node = self.nodes.EmptyLatentImage()
        self._sampler_node = self.nodes.KSampler()
        self._decode_node = self.nodes.VAEDecode()

    def _validate_root(self) -> None:
        if not self.root.is_dir() or not (self.root / "nodes.py").is_file():
            raise RuntimeError(
                f"COMFYUI_PATH must point to a local ComfyUI source checkout containing nodes.py: {self.root}"
            )

    def patch_aura(self, model: Any, shift: float) -> Any:
        return self._sampling_node.patch_aura(model, shift)[0]

    def encode_edit(self, clip: Any, prompt: str, vae: Any, image1: Any, image2: Any = None, image3: Any = None) -> Any:
        with self.torch.inference_mode():
            output = self._qwen_encode_node.execute(
                clip=clip,
                prompt=prompt,
                vae=vae,
                image1=image1,
                image2=image2,
                image3=image3,
            )
        return output[0]

    def encode_negative(self, clip: Any, text: str) -> Any:
        with self.torch.inference_mode():
            return self._negative_encode_node.encode(clip, text)

    def empty_latent(self, width: int, height: int, batch_size: int) -> Any:
        return self._latent_node.generate(width, height, batch_size)

    def sample(self, **kwargs: Any) -> Any:
        with self.torch.inference_mode():
            return self._sampler_node.sample(**kwargs)

    def decode(self, vae: Any, samples: Any) -> Any:
        with self.torch.inference_mode():
            return self._decode_node.decode(vae, samples)

    def cleanup(self) -> None:
        soft_empty_cache = getattr(self.model_management, "soft_empty_cache", None)
        if callable(soft_empty_cache):
            soft_empty_cache()


class ComfyUIEditBackend:
    """Qwen Image Edit using ComfyUI-native component checkpoints and sampling."""

    def __init__(self, settings: Settings, *, runtime: Optional[Any] = None) -> None:
        self.settings = settings
        self._runtime = runtime
        self._model = None
        self._clip = None
        self._vae = None
        self._component_key: Optional[tuple[Any, ...]] = None
        self._lock = threading.Lock()

    def prepare(
        self,
        *,
        lora_path: Optional[str] = None,
        lora_weight_name: Optional[str] = None,
        lora_scale: float = 1.0,
    ) -> None:
        with self._lock:
            self._load_components(lora_path=lora_path, lora_weight_name=lora_weight_name, lora_scale=lora_scale)

    def edit(
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
        with self._lock:
            runtime = self._get_runtime()
            model, clip, vae = self._load_components(
                lora_path=lora_path,
                lora_weight_name=lora_weight_name,
                lora_scale=lora_scale,
            )
            images = image if isinstance(image, list) else [image]
            tensors = [self._pil_to_tensor(item) for item in images[:3]]
            positive = runtime.encode_edit(
                clip,
                prompt,
                vae,
                tensors[0],
                tensors[1] if len(tensors) > 1 else None,
                tensors[2] if len(tensors) > 2 else None,
            )
            negative_text = self.settings.comfyui_qwen_edit_negative_prompt.strip() or " "
            negative = runtime.encode_negative(clip, negative_text)[0]
            # Use input image dimensions for latent to match ComfyUI workflow behavior
            img_width, img_height = images[0].size
            latent = runtime.empty_latent(img_width, img_height, 1)[0]
            samples = runtime.sample(
                model=model,
                seed=seed if seed is not None else self.settings.comfyui_qwen_edit_default_seed,
                steps=num_inference_steps,
                cfg=guidance_scale,
                sampler_name=self.settings.comfyui_qwen_edit_sampler_name,
                scheduler=self.settings.comfyui_qwen_edit_scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=1.0 if strength is None else strength,
            )[0]
            decoded = runtime.decode(vae, samples)[0]
            return self._tensor_to_pil(decoded)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._clip = None
            self._vae = None
            self._component_key = None
            if self._runtime is not None:
                self._runtime.cleanup()

    def _load_components(
        self,
        *,
        lora_path: Optional[str],
        lora_weight_name: Optional[str],
        lora_scale: float,
    ) -> tuple[Any, Any, Any]:
        runtime = self._get_runtime()
        loras = self._resolve_loras(lora_path, lora_weight_name, lora_scale)
        key = (
            self.settings.comfyui_qwen_edit_unet_name,
            self.settings.comfyui_qwen_edit_unet_weight_dtype,
            self.settings.comfyui_qwen_edit_clip_name,
            self.settings.comfyui_qwen_edit_clip_device,
            self.settings.comfyui_qwen_edit_vae_name,
            self.settings.comfyui_qwen_edit_model_shift,
            tuple(loras),
        )
        if key == self._component_key and self._model is not None:
            return self._model, self._clip, self._vae

        self._register_model_paths(runtime)
        model_options: dict[str, Any] = {}
        weight_dtype = self.settings.comfyui_qwen_edit_unet_weight_dtype
        if weight_dtype in {"fp8_e4m3fn", "fp8_e4m3fn_fast"}:
            model_options["dtype"] = runtime.torch.float8_e4m3fn
            if weight_dtype.endswith("_fast"):
                model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = runtime.torch.float8_e5m2

        unet_path = runtime.folder_paths.get_full_path_or_raise(
            "diffusion_models", self.settings.comfyui_qwen_edit_unet_name
        )
        model = runtime.sd.load_diffusion_model(unet_path, model_options=model_options)

        clip_options: dict[str, Any] = {}
        if self.settings.comfyui_qwen_edit_clip_device == "cpu":
            cpu = runtime.torch.device("cpu")
            clip_options["load_device"] = cpu
            clip_options["offload_device"] = cpu
        clip_path = runtime.folder_paths.get_full_path_or_raise(
            "text_encoders", self.settings.comfyui_qwen_edit_clip_name
        )
        clip = runtime.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=runtime.folder_paths.get_folder_paths("embeddings"),
            clip_type=runtime.sd.CLIPType.QWEN_IMAGE,
            model_options=clip_options,
        )

        vae_path = runtime.folder_paths.get_full_path_or_raise("vae", self.settings.comfyui_qwen_edit_vae_name)
        vae_state, vae_metadata = runtime.utils.load_torch_file(vae_path, return_metadata=True)
        vae = runtime.sd.VAE(sd=vae_state, metadata=vae_metadata)
        validate_vae = getattr(vae, "throw_exception_if_invalid", None)
        if callable(validate_vae):
            validate_vae()

        for lora_name, scale in loras:
            lora_file = runtime.folder_paths.get_full_path_or_raise("loras", lora_name)
            lora, metadata = runtime.utils.load_torch_file(lora_file, safe_load=True, return_metadata=True)
            model, _ = runtime.sd.load_lora_for_models(
                model,
                None,
                lora,
                scale,
                0,
                lora_metadata=metadata,
            )

        model = runtime.patch_aura(model, self.settings.comfyui_qwen_edit_model_shift)
        self._model, self._clip, self._vae = model, clip, vae
        self._component_key = key
        logger.info(
            "Loaded ComfyUI Qwen Edit components: unet=%s clip=%s vae=%s loras=%s",
            self.settings.comfyui_qwen_edit_unet_name,
            self.settings.comfyui_qwen_edit_clip_name,
            self.settings.comfyui_qwen_edit_vae_name,
            [name for name, _ in loras],
        )
        return model, clip, vae

    def _resolve_loras(
        self,
        lora_path: Optional[str],
        lora_weight_name: Optional[str],
        lora_scale: float,
    ) -> list[tuple[str, float]]:
        loras: list[tuple[str, float]] = []
        if self.settings.qwen_edit_lightning_lora_enabled and self.settings.comfyui_qwen_edit_lora_name:
            loras.append((self.settings.comfyui_qwen_edit_lora_name, self.settings.qwen_edit_lightning_lora_scale))
        if lora_path:
            name = lora_weight_name or Path(lora_path).name
            if name and all(existing_name != name for existing_name, _ in loras):
                loras.append((name, lora_scale))
        return loras

    def _register_model_paths(self, runtime: Any) -> None:
        models = self.settings.comfyui_models_path.expanduser().resolve()
        for category, folder in (
            ("diffusion_models", "diffusion_models"),
            ("text_encoders", "text_encoders"),
            ("vae", "vae"),
            ("loras", "loras"),
            ("embeddings", "embeddings"),
        ):
            path = models / folder
            add_path = runtime.folder_paths.add_model_folder_path
            try:
                add_path(category, str(path), is_default=True)
            except TypeError:
                add_path(category, str(path))

    def _get_runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = ComfyUIRuntime(self.settings.comfyui_path)
        return self._runtime

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> Any:
        import numpy as np
        import torch

        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    @staticmethod
    def _tensor_to_pil(tensor: Any) -> Image.Image:
        import numpy as np

        array = tensor[0].detach().float().cpu().clamp(0, 1).numpy()
        return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")
import logging
from typing import Any

from app.config import Settings


logger = logging.getLogger(__name__)


def get_pipeline_device_map_kwargs(settings: Settings, torch: Any, device: str) -> dict[str, Any]:
    if not device.startswith("cuda") or settings.model_gpu_count <= 1:
        return {}

    available_gpu_count = torch.cuda.device_count()
    gpu_count = min(settings.model_gpu_count, available_gpu_count, 4)
    if gpu_count <= 1:
        return {}

    kwargs: dict[str, Any] = {"device_map": "balanced"}
    max_memory = _device_map_max_memory(settings, torch, gpu_count)
    if max_memory:
        kwargs["max_memory"] = max_memory
    logger.info(
        "Enabled multi-GPU pipeline loading: device_map=balanced gpu_count=%s visible_devices=%s",
        gpu_count,
        settings.model_gpu_ids or "CUDA_VISIBLE_DEVICES/default",
    )
    return kwargs


def uses_pipeline_device_map(load_kwargs: dict[str, Any]) -> bool:
    return "device_map" in load_kwargs


def apply_pipeline_cpu_offload(pipe: object, settings: Settings, device: str) -> bool:
    if not settings.enable_cpu_offload or not device.startswith("cuda"):
        return False

    if settings.cpu_offload_mode == "sequential":
        if _call_first_available(pipe, ("enable_sequential_cpu_offload",)):
            logger.info("Enabled sequential CPU offload for pipeline")
            return True

        logger.warning(
            "Pipeline does not support sequential CPU offload; falling back to model CPU offload"
        )

    if _call_first_available(pipe, ("enable_model_cpu_offload",)):
        logger.info("Enabled model CPU offload for pipeline")
        return True

    if settings.cpu_offload_mode == "model" and _call_first_available(pipe, ("enable_sequential_cpu_offload",)):
        logger.warning(
            "Pipeline does not support model CPU offload; falling back to slower sequential CPU offload"
        )
        return True

    logger.warning("CPU offload requested, but pipeline does not expose an offload method")
    return False


def _device_map_max_memory(settings: Settings, torch: Any, gpu_count: int) -> dict[int, str]:
    if settings.model_gpu_memory_limit:
        return {index: settings.model_gpu_memory_limit for index in range(gpu_count)}

    remaining_max_memory = get_remaining_cuda_max_memory(settings, torch, reserve_gib=2)
    if remaining_max_memory:
        return remaining_max_memory

    max_memory: dict[int, str] = {}
    for index in range(gpu_count):
        try:
            total_gib = torch.cuda.get_device_properties(index).total_memory // (1024**3)
        except Exception as exc:
            logger.debug("Failed to read CUDA device %s memory: %s", index, exc)
            return {}
        if total_gib > 0:
            max_memory[index] = f"{total_gib}GiB"
    return max_memory


def get_remaining_cuda_max_memory(settings: Settings, torch: Any, reserve_gib: int = 2) -> dict[int, str]:
    try:
        if not torch.cuda.is_available():
            return {}
    except Exception as exc:
        logger.debug("Failed to query CUDA availability: %s", exc)
        return {}

    available_gpu_count = torch.cuda.device_count()
    gpu_count = min(settings.model_gpu_count, available_gpu_count, 4)
    if gpu_count <= 0:
        return {}

    max_memory: dict[int, str] = {}
    for index in range(gpu_count):
        try:
            with torch.cuda.device(index):
                free_bytes, _ = torch.cuda.mem_get_info()
            # 按当前剩余显存给 device_map 上限，避免前一阶段已占显存后仍按总显存继续塞到同一张卡。
            free_gib = max(1, int(free_bytes // (1024**3)) - reserve_gib)
            max_memory[index] = f"{free_gib}GiB"
        except Exception as exc:
            logger.debug("Failed to read remaining CUDA memory for cuda:%s: %s", index, exc)
            return {}
    return max_memory


def apply_pipeline_memory_settings(pipe: object, settings: Settings) -> None:
    if settings.enable_vae_tiling:
        _call_vae_method(pipe, "enable_tiling") or _call_first_available(pipe, ("enable_vae_tiling",))
    else:
        _call_vae_method(pipe, "disable_tiling") or _call_first_available(pipe, ("disable_vae_tiling",))

    if settings.enable_vae_slicing:
        _call_vae_method(pipe, "enable_slicing") or _call_first_available(pipe, ("enable_vae_slicing",))
    else:
        _call_vae_method(pipe, "disable_slicing") or _call_first_available(pipe, ("disable_vae_slicing",))

    if settings.enable_attention_slicing:
        _call_first_available(pipe, ("enable_attention_slicing",))
    else:
        _call_first_available(pipe, ("disable_attention_slicing",))


def _call_first_available(target: object, method_names: tuple[str, ...]) -> bool:
    for method_name in method_names:
        method = getattr(target, method_name, None)
        if not callable(method):
            continue
        try:
            method()
            return True
        except TypeError:
            try:
                method(None)
                return True
            except Exception as exc:
                logger.debug("Failed to call %s(None): %s", method_name, exc)
        except Exception as exc:
            logger.debug("Failed to call %s(): %s", method_name, exc)
    return False


def _call_vae_method(pipe: object, method_name: str) -> bool:
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return False
    return _call_first_available(vae, (method_name,))
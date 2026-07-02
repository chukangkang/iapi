import logging

from app.config import Settings


logger = logging.getLogger(__name__)


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
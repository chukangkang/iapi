from functools import lru_cache
import math
import os
from pathlib import Path
import socket
from typing import Literal, Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = "flux-image-backend"
    model_path: str = "black-forest-labs/FLUX.2-klein-9b-kv"
    device: str = "auto"
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "bfloat16"
    num_inference_steps: int = Field(default=4, ge=1)
    default_width: int = Field(default=1328, ge=64)
    default_height: int = Field(default=1328, ge=64)
    max_generation_pixels: int = Field(default=786432, ge=65536)
    response_metadata_enabled: bool = True
    default_negative_prompt: str = "low resolution, low quality, deformed limbs, deformed fingers, oversaturated image, wax figure look, face lacking details, overly smooth skin, smooth skin, plastic look, blurry, oil painting look, AI-generated look, chaotic composition, blurry text, distorted text"
    prompt_enhance_enabled: bool = True
    prompt_enhance_short_max_chars: int = Field(default=80, ge=1)
    prompt_enhance_suffix: str = "high quality, detailed, natural lighting, realistic, clean composition"
    prompt_enhance_person_suffix: str = ""
    default_enhance_mode: Literal["flux", "qwen_image", "pixel", "realesrgan", "realesrgan_flux", "qwen_edit", "qwen_edit_realesrgan", "qwen_unblur_upscale", "qwen_unblur_upscale_realesrgan"] = "flux"
    flux_refine_strength: float = Field(default=0.08, ge=0.0, le=1.0)
    qwen_image_model_name: str = "qwen-image-2512"
    qwen_image_model_path: str = "unsloth/Qwen-Image-2512-unsloth-bnb-4bit"
    qwen_image_pipeline_class: str = "QwenImagePipeline"
    qwen_image_lora_path: str = "Wuli-art/Qwen-Image-2512-Turbo-LoRA-2-Steps"
    qwen_image_lora_weight_name: str = ""
    qwen_image_lora_adapter_name: str = "qwen_image_2512_turbo"
    qwen_image_lora_scale: float = Field(default=1.0, ge=0.0)
    qwen_image_steps: int = Field(default=2, ge=1)
    qwen_image_guidance_scale: float = Field(default=1.0, ge=0.0)
    qwen_image_true_cfg_scale: float = Field(default=1.0, ge=0.0)
    qwen_image_scheduler_exponential_shift_mu: float = Field(default=math.log(2.5), ge=0.0)
    qwen_image_scheduler_use_dynamic_shifting: bool = True
    qwen_image_scheduler_shift_terminal: float = Field(default=0.7155, ge=0.0)
    qwen_edit_model_path: str = "Qwen/Qwen-Image-Edit-2511"
    qwen_edit_pipeline_class: str = "QwenImageEditPlusPipeline"
    qwen_edit_steps: int = Field(default=10, ge=1)
    qwen_edit_guidance_scale: float = Field(default=1.0, ge=0.0)
    qwen_edit_true_cfg_scale: float = Field(default=4.0, ge=0.0)
    qwen_edit_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    qwen_edit_scale_to_side: Literal["longest", "shortest"] = "longest"
    qwen_edit_scale_to_length: int = Field(default=2048, ge=64)
    qwen_edit_round_to_multiple: int = Field(default=16, ge=1)
    qwen_edit_input_fit_mode: Literal["contain", "cover"] = "contain"
    qwen_edit_background_color: str = "#000000"
    qwen_edit_quantization: Literal["none", "8bit", "4bit"] = "none"
    qwen_edit_device_map: Literal["balanced", "cuda", "cpu"] = "balanced"
    qwen_unblur_upscale_lora_path: str = "prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale"
    qwen_unblur_upscale_lora_weight_name: str = "Qwen-Image-Edit-Unblur-Upscale_20.safetensors"
    qwen_unblur_upscale_trigger_prompt: str = "unblur and upscale"
    qwen_unblur_upscale_lora_scale: float = Field(default=1.0, ge=0.0)
    pixel_sharpen_enabled: bool = True
    pixel_sharpen_radius: float = Field(default=1.4, ge=0.0)
    pixel_sharpen_percent: int = Field(default=140, ge=0)
    pixel_sharpen_threshold: int = Field(default=3, ge=0)
    upscale_fit_mode: Literal["stretch", "contain", "cover"] = "cover"
    upscale_fill_color: str = "black"
    realesrgan_model_path: str = ""
    realesrgan_model_name: str = "realesr-general-x4v3"
    realesrgan_max_passes: int = Field(default=2, ge=1, le=4)
    realesrgan_denoise_strength: float = Field(default=0.35, ge=0.0, le=1.0)
    realesrgan_outscale: float = Field(default=0.0, ge=0.0)
    realesrgan_tile: int = Field(default=512, ge=0)
    realesrgan_tile_pad: int = Field(default=10, ge=0)
    realesrgan_pre_pad: int = Field(default=0, ge=0)
    realesrgan_face_enhance: bool = False
    realesrgan_fp32: bool = False
    realesrgan_gpu_id: Optional[int] = None
    realesrgan_alpha_upsampler: Literal["realesrgan", "bicubic"] = "realesrgan"
    image_worker_count: int = Field(default=1, ge=1)
    image_queue_maxsize: int = Field(default=100, ge=1)
    worker_name: str = ""
    service_role: Literal["api", "worker", "all"] = "all"
    task_queue_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_queue_name: str = "iapi:image_tasks"
    redis_processing_queue_name: str = "iapi:image_tasks:processing"
    redis_block_timeout: int = Field(default=5, ge=1)
    redis_socket_connect_timeout: int = Field(default=10, ge=1)
    redis_socket_timeout: int = Field(default=30, ge=1)
    redis_requeue_stale_enabled: bool = True
    redis_processing_timeout: int = Field(default=90, ge=1)
    redis_requeue_interval: int = Field(default=30, ge=1)
    task_running_heartbeat_interval: int = Field(default=30, ge=1)
    task_db_backend: Literal["sqlite", "mysql"] = "sqlite"
    task_db_path: Path = Path("data/image_tasks.sqlite3")
    mysql_host: str = "127.0.0.1"
    mysql_port: int = Field(default=3306, ge=1, le=65535)
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "iapi"
    mysql_charset: str = "utf8mb4"
    mysql_connect_timeout: int = Field(default=10, ge=1)
    task_public_base_url: str = ""
    output_dir: Path = Path("outputs")
    public_base_url: str = ""
    hf_token: str = ""
    enable_cpu_offload: bool = False
    cpu_offload_mode: Literal["model", "sequential"] = "model"
    enable_vae_tiling: bool = True
    enable_vae_slicing: bool = True
    enable_attention_slicing: bool = False
    pytorch_cuda_alloc_conf: str = "expandable_segments:True"
    tokenizers_parallelism: str = "false"
    oss_endpoint: str = ""
    oss_access_key_id: str = ""
    oss_access_key_secret: str = ""
    oss_bucket: str = ""
    oss_public_base_url: str = ""
    oss_object_prefix: str = "images"
    oss_retention_days: int = Field(default=14, ge=1)
    oss_sts_role_arn: str = ""
    oss_sts_duration: int = Field(default=3600, ge=900, le=43200)
    aliyun_region_id: str = "cn-hangzhou"

    @computed_field
    @property
    def oss_enabled(self) -> bool:
        return all([self.oss_endpoint, self.oss_access_key_id, self.oss_access_key_secret, self.oss_bucket])

    @computed_field
    @property
    def normalized_oss_public_base_url(self) -> str:
        return self.oss_public_base_url.rstrip("/")

    @computed_field
    @property
    def normalized_public_base_url(self) -> str:
        return self.public_base_url.rstrip("/")

    @computed_field
    @property
    def normalized_task_public_base_url(self) -> str:
        return self.task_public_base_url.rstrip("/")

    @computed_field
    @property
    def mysql_password_or_none(self) -> Optional[str]:
        return self.mysql_password or None

    @computed_field
    @property
    def resolved_worker_name(self) -> str:
        return self.worker_name.strip() or socket.gethostname()


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.pytorch_cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", settings.pytorch_cuda_alloc_conf)
    if settings.tokenizers_parallelism:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", settings.tokenizers_parallelism)
    return settings

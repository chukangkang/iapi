from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

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
    default_width: int = Field(default=768, ge=64)
    default_height: int = Field(default=768, ge=64)
    max_generation_pixels: int = Field(default=786432, ge=65536)
    default_enhance_mode: Literal["flux", "pixel", "realesrgan", "realesrgan_flux", "qwen_edit", "qwen_edit_realesrgan"] = "flux"
    flux_refine_strength: float = Field(default=0.08, ge=0.0, le=1.0)
    qwen_edit_model_path: str = "Qwen/Qwen-Image-Edit"
    qwen_edit_pipeline_class: str = "QwenImageEditPipeline"
    qwen_edit_steps: int = Field(default=10, ge=1)
    qwen_edit_guidance_scale: float = Field(default=1.0, ge=0.0)
    qwen_edit_true_cfg_scale: float = Field(default=4.0, ge=0.0)
    qwen_edit_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    qwen_edit_max_pixels: int = Field(default=1048576, ge=65536)
    qwen_edit_scale_to_side: Literal["longest", "shortest"] = "longest"
    qwen_edit_scale_to_length: int = Field(default=2048, ge=64)
    qwen_edit_round_to_multiple: int = Field(default=16, ge=1)
    qwen_edit_background_color: str = "#000000"
    qwen_edit_quantization: Literal["none", "8bit", "4bit"] = "none"
    qwen_edit_device_map: Literal["balanced", "cuda", "cpu"] = "balanced"
    pixel_sharpen_enabled: bool = True
    pixel_sharpen_radius: float = Field(default=1.4, ge=0.0)
    pixel_sharpen_percent: int = Field(default=140, ge=0)
    pixel_sharpen_threshold: int = Field(default=3, ge=0)
    upscale_fit_mode: Literal["stretch", "contain", "cover"] = "cover"
    upscale_fill_color: str = "black"
    realesrgan_model_path: str = ""
    realesrgan_model_name: str = "realesr-general-x4v3.pth"
    realesrgan_max_passes: int = Field(default=2, ge=1, le=4)
    realesrgan_denoise_strength: float = Field(default=0.35, ge=0.0, le=1.0)
    realesrgan_tile: int = Field(default=512, ge=0)
    realesrgan_tile_pad: int = Field(default=10, ge=0)
    realesrgan_pre_pad: int = Field(default=0, ge=0)
    image_worker_count: int = Field(default=1, ge=1)
    image_queue_maxsize: int = Field(default=100, ge=1)
    task_db_path: Path = Path("data/image_tasks.sqlite3")
    task_public_base_url: str = ""
    output_dir: Path = Path("outputs")
    public_base_url: str = ""
    hf_token: str = ""
    enable_cpu_offload: bool = True
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


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.pytorch_cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", settings.pytorch_cuda_alloc_conf)
    if settings.tokenizers_parallelism:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", settings.tokenizers_parallelism)
    return settings

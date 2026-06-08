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
    image_worker_count: int = Field(default=1, ge=1)
    image_queue_maxsize: int = Field(default=100, ge=1)
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

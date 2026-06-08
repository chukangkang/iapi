from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "black-forest-labs/FLUX.2-klein-9b-kv"
    device: str = "auto"
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "bfloat16"
    num_inference_steps: int = Field(default=4, ge=1)
    default_width: int = Field(default=1024, ge=64)
    default_height: int = Field(default=1024, ge=64)
    output_dir: Path = Path("outputs")
    public_base_url: str = ""
    hf_token: str = ""
    enable_cpu_offload: bool = False

    @computed_field
    @property
    def normalized_public_base_url(self) -> str:
        return self.public_base_url.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    return Settings()

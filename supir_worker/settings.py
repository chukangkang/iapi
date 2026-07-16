from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SupirWorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUPIR_WORKER_",
        env_file=".env.supir-worker",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = Field(default=8010, ge=1, le=65535)
    api_key: str = ""
    preload: bool = True
    repo_path: Path = Path("/opt/SUPIR")
    config_path: Path = Path("/opt/SUPIR/options/SUPIR_v0.yaml")
    sdxl_checkpoint: Path = Path("/models/sd_xl_base_1.0_0.9vae.safetensors")
    supir_q_checkpoint: Path = Path("/models/SUPIR-v0Q.ckpt")
    supir_f_checkpoint: Path = Path("/models/SUPIR-v0F.ckpt")
    model_sign: Literal["Q", "F"] = "Q"
    device: str = "cuda:0"
    ae_dtype: Literal["fp32", "bf16"] = "bf16"
    diffusion_dtype: Literal["fp32", "fp16", "bf16"] = "fp16"
    loading_half_params: bool = True
    use_tile_vae: bool = True
    encoder_tile_size: int = Field(default=512, ge=64)
    decoder_tile_size: int = Field(default=64, ge=16)
    min_size: int = Field(default=1024, ge=64)
    edm_steps: int = Field(default=50, ge=1)
    seed: int = 1234
    s_stage1: float = -1.0
    s_churn: float = 5.0
    s_noise: float = 1.01
    s_cfg: float = 4.0
    s_stage2: float = 1.0
    color_fix_type: Literal["None", "AdaIn", "Wavelet"] = "Wavelet"
    positive_prompt: str = (
        "Cinematic, highly detailed, realistic photograph, natural texture, "
        "sharp focus, perfect without deformations."
    )
    negative_prompt: str = (
        "painting, illustration, cartoon, 3D render, blurry, low quality, "
        "watermark, jpeg artifacts, deformed, over-smooth"
    )
    max_pixels: int = Field(default=16777216, ge=65536)
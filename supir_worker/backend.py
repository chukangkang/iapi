import logging
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from supir_worker.settings import SupirWorkerSettings


logger = logging.getLogger(__name__)


class SupirBackend:
    """Lazy, serialized adapter around the official SUPIR inference API."""

    def __init__(self, settings: SupirWorkerSettings):
        self.settings = settings
        self.model_sign = settings.model_sign
        self.ready = False
        self._model: Any = None
        self._torch: Any = None
        self._pil_to_tensor: Any = None
        self._tensor_to_pil: Any = None
        self._lock = threading.Lock()
        self._runtime_config: Path | None = None

    def load(self) -> None:
        if self.ready:
            return
        with self._lock:
            if self.ready:
                return
            self._validate_files()
            repo_path = str(self.settings.repo_path.resolve())
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
            import torch
            from omegaconf import OmegaConf
            from SUPIR.util import PIL2Tensor, Tensor2PIL, convert_dtype, create_SUPIR_model

            if not torch.cuda.is_available():
                raise RuntimeError("The official SUPIR backend requires an NVIDIA CUDA GPU")
            config = OmegaConf.load(str(self.settings.config_path))
            config.SDXL_CKPT = str(self.settings.sdxl_checkpoint)
            config.SUPIR_CKPT_Q = str(self.settings.supir_q_checkpoint)
            config.SUPIR_CKPT_F = str(self.settings.supir_f_checkpoint)
            runtime = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
            runtime.close()
            self._runtime_config = Path(runtime.name)
            OmegaConf.save(config, str(self._runtime_config))

            model = create_SUPIR_model(str(self._runtime_config), SUPIR_sign=self.model_sign)
            if self.settings.loading_half_params:
                model = model.half()
            if self.settings.use_tile_vae:
                model.init_tile_vae(
                    encoder_tile_size=self.settings.encoder_tile_size,
                    decoder_tile_size=self.settings.decoder_tile_size,
                )
            model.ae_dtype = convert_dtype(self.settings.ae_dtype)
            model.model.dtype = convert_dtype(self.settings.diffusion_dtype)
            self._model = model.to(self.settings.device)
            self._torch = torch
            self._pil_to_tensor = PIL2Tensor
            self._tensor_to_pil = Tensor2PIL
            self.ready = True
            logger.info("SUPIR-%s loaded on %s", self.model_sign, self.settings.device)

    def restore(self, image: Image.Image, *, prompt: str, width: int, height: int) -> Image.Image:
        self.load()
        with self._lock:
            prepared = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            tensor, output_height, output_width = self._pil_to_tensor(
                prepared,
                upsacle=1,
                min_size=self.settings.min_size,
            )
            tensor = tensor.unsqueeze(0).to(self.settings.device)[:, :3, :, :]
            captions = [prompt.strip()]
            with self._torch.inference_mode():
                samples = self._model.batchify_sample(
                    tensor,
                    captions,
                    num_steps=self.settings.edm_steps,
                    restoration_scale=self.settings.s_stage1,
                    s_churn=self.settings.s_churn,
                    s_noise=self.settings.s_noise,
                    cfg_scale=self.settings.s_cfg,
                    control_scale=self.settings.s_stage2,
                    seed=self.settings.seed,
                    num_samples=1,
                    p_p=self.settings.positive_prompt,
                    n_p=self.settings.negative_prompt,
                    color_fix_type=self.settings.color_fix_type,
                    use_linear_CFG=True,
                    use_linear_control_scale=False,
                    cfg_scale_start=1.0,
                    control_scale_start=0.0,
                )
            restored = self._tensor_to_pil(samples[0], output_height, output_width)
            return restored.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")

    def _validate_files(self) -> None:
        required = [
            self.settings.config_path,
            self.settings.sdxl_checkpoint,
            self.settings.supir_q_checkpoint if self.model_sign == "Q" else self.settings.supir_f_checkpoint,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing SUPIR model/config files: " + ", ".join(missing))
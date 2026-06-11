import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

from app.config import Settings

SEEDVR_REQUIRED_SOURCE_PATHS = (
    "projects/inference_seedvr2_3b.py",
    "configs_3b/main.yaml",
    "common",
    "models",
    "data/image/transforms/divisible_crop.py",
    "data/image/transforms/na_resize.py",
    "data/video/transforms/rearrange.py",
    "projects/video_diffusion_sr",
    "pos_emb.pt",
    "neg_emb.pt",
)
SEEDVR_SOURCE_MODULE_PREFIXES = ("common", "configs_3b", "data", "models", "projects")
SEEDVR_PACKAGE_DIRS = (
    "common",
    "configs_3b",
    "data",
    "data/image",
    "data/image/transforms",
    "data/video",
    "data/video/transforms",
    "models",
    "projects",
    "projects/video_diffusion_sr",
)


class SeedVR2Service:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def enhance(
        self,
        image: Image.Image,
        *,
        width: int,
        height: int,
        seed: Optional[int],
    ) -> Image.Image:
        return await asyncio.to_thread(self._enhance_sync, image, width=width, height=height, seed=seed)

    def _enhance_sync(self, image: Image.Image, *, width: int, height: int, seed: Optional[int]) -> Image.Image:
        repo_path, script_path, model_path, vae_path = self._validate_paths()
        with tempfile.TemporaryDirectory(prefix="seedvr2_iapi_") as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            input_path = input_dir / "input.png"
            image.convert("RGB").save(input_path)
            self._ensure_package_markers(repo_path)
            self._prepare_checkpoints(repo_path, model_path, vae_path)
            self._run_official_script(
                repo_path=repo_path,
                script_path=script_path,
                input_dir=input_dir,
                output_dir=output_dir,
                width=width,
                height=height,
                seed=seed,
            )
            output_path = self._find_output_image(output_dir, input_path.name)
            with Image.open(output_path) as output_image:
                return output_image.convert("RGB").copy()

    def _validate_paths(self) -> tuple[Path, Path, Path, Path]:
        repo_path = self.settings.seedvr2_repo_path.strip()
        if not repo_path:
            raise RuntimeError(
                "SEEDVR2_REPO_PATH is required for enhance_mode=seedvr2 or qwen_edit_seedvr2. "
                "It must point to the cloned GitHub code repo https://github.com/ByteDance-Seed/SeedVR, "
                "not the Hugging Face SeedVR2-3B weights folder."
            )
        if not self.settings.seedvr2_model_path.strip():
            raise RuntimeError("SEEDVR2_MODEL_PATH is required for enhance_mode=seedvr2 or qwen_edit_seedvr2.")
        if not self.settings.seedvr2_vae_path.strip():
            raise RuntimeError("SEEDVR2_VAE_PATH is required for enhance_mode=seedvr2 or qwen_edit_seedvr2.")

        resolved_repo_path = Path(repo_path)
        if not resolved_repo_path.exists():
            raise FileNotFoundError(f"SeedVR2 repo path not found: {resolved_repo_path}")
        script_path = resolved_repo_path / "projects" / "inference_seedvr2_3b.py"
        if not script_path.exists():
            raise FileNotFoundError(
                "SeedVR2 official inference script not found. "
                f"Expected: {script_path}. "
                "SEEDVR2_REPO_PATH must point to a clone of https://github.com/ByteDance-Seed/SeedVR. "
                "The Hugging Face ByteDance-Seed/SeedVR2-3B repository only contains weights."
            )
        missing_source_paths = [relative_path for relative_path in SEEDVR_REQUIRED_SOURCE_PATHS if not (resolved_repo_path / relative_path).exists()]
        if missing_source_paths:
            raise FileNotFoundError(
                "SeedVR2 repo checkout is incomplete or SEEDVR2_REPO_PATH points to the wrong directory. "
                f"Repo path: {resolved_repo_path}. Missing: {', '.join(missing_source_paths)}. "
                "Run scripts/download_seedvr.sh on the server from the iAPI project root, then set SEEDVR2_REPO_PATH to that SeedVR directory."
            )

        model_path = Path(self.settings.seedvr2_model_path)
        vae_path = Path(self.settings.seedvr2_vae_path)
        for checkpoint_path in [model_path, vae_path]:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"SeedVR2 checkpoint not found: {checkpoint_path}")
            if checkpoint_path.suffix.lower() != ".pth":
                raise ValueError(
                    "The official SeedVR2 inference script loads PyTorch .pth checkpoints. "
                    f"Unsupported checkpoint format: {checkpoint_path}. "
                    "Use seedvr2_ema_3b.pth and ema_vae.pth from ByteDance-Seed/SeedVR2-3B."
                )
        return resolved_repo_path, script_path, model_path, vae_path

    def _ensure_package_markers(self, repo_path: Path) -> None:
        for relative_dir in SEEDVR_PACKAGE_DIRS:
            package_dir = repo_path / relative_dir
            if package_dir.exists() and package_dir.is_dir():
                init_path = package_dir / "__init__.py"
                init_path.touch(exist_ok=True)

    def _prepare_checkpoints(self, repo_path: Path, model_path: Path, vae_path: Path) -> None:
        ckpt_dir = repo_path / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_checkpoint_link_or_copy(model_path, ckpt_dir / "seedvr2_ema_3b.pth")
        self._ensure_checkpoint_link_or_copy(vae_path, ckpt_dir / "ema_vae.pth")

    def _ensure_checkpoint_link_or_copy(self, source_path: Path, target_path: Path) -> None:
        if target_path.exists():
            if target_path.resolve() == source_path.resolve():
                return
            target_path.unlink()
        try:
            target_path.symlink_to(source_path.resolve())
        except OSError:
            shutil.copy2(source_path, target_path)

    def _run_official_script(
        self,
        *,
        repo_path: Path,
        script_path: Path,
        input_dir: Path,
        output_dir: Path,
        width: int,
        height: int,
        seed: Optional[int],
    ) -> None:
        env = os.environ.copy()
        python_path = Path(self.settings.seedvr2_python or sys.executable).resolve()
        if not python_path.exists():
            raise FileNotFoundError(f"SeedVR2 Python executable not found: {python_path}")
        python_bin_dir = python_path.parent
        env_prefix = python_bin_dir.parent
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(repo_path) if not existing_pythonpath else f"{repo_path}{os.pathsep}{existing_pythonpath}"
        existing_path = env.get("PATH", "")
        env["PATH"] = str(python_bin_dir) if not existing_path else f"{python_bin_dir}{os.pathsep}{existing_path}"
        env.setdefault("VIRTUAL_ENV", str(env_prefix))
        env.setdefault("CONDA_PREFIX", str(env_prefix))
        env.setdefault("PYTHONNOUSERSITE", "1")
        command = [
            str(python_path),
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=1",
            str(script_path.relative_to(repo_path)),
            "--video_path",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--seed",
            str(seed if seed is not None else 666),
            "--res_h",
            str(height),
            "--res_w",
            str(width),
            "--sp_size",
            "1",
        ]
        completed = subprocess.run(
            command,
            cwd=repo_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stdout = completed.stdout[-4000:] if completed.stdout else ""
            stderr = completed.stderr[-4000:] if completed.stderr else ""
            missing_module_match = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", stderr)
            install_hint = ""
            if missing_module_match:
                missing_module = missing_module_match.group(1)
                if missing_module.partition(".")[0] in SEEDVR_SOURCE_MODULE_PREFIXES:
                    install_hint = (
                        f"\nMissing SeedVR source module: {missing_module}. "
                        f"SEEDVR2_REPO_PATH may be incomplete or wrong: {repo_path}. "
                        "Run scripts/download_seedvr.sh on the server from the iAPI project root, "
                        "and ensure data/image/transforms/divisible_crop.py exists under SEEDVR2_REPO_PATH."
                    )
                else:
                    install_hint = (
                        f"\nMissing Python package: {missing_module}. "
                        "Install SeedVR dependencies in the environment used by SEEDVR2_PYTHON: "
                        f"{python_path} -m pip install -r requirements-seedvr.txt"
                    )
            raise RuntimeError(
                "SeedVR2 official inference failed. "
                f"Command: {' '.join(command)}\nPython: {python_path}\nEnv prefix: {env_prefix}{install_hint}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )

    def _find_output_image(self, output_dir: Path, input_filename: str) -> Path:
        preferred_path = output_dir / input_filename
        if preferred_path.exists():
            return preferred_path
        image_paths = sorted(
            path for path in output_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        )
        if image_paths:
            return image_paths[0]
        raise FileNotFoundError(f"SeedVR2 did not write an output image to {output_dir}")


def seedvr2_config_error(settings: Settings) -> Optional[str]:
    if not settings.seedvr2_repo_path.strip():
        return "SEEDVR2_REPO_PATH is required for SeedVR2 modes and must point to the cloned GitHub SeedVR code repo, not the HF weights folder."
    if not settings.seedvr2_model_path.strip():
        return "SEEDVR2_MODEL_PATH is required for SeedVR2 modes."
    if not settings.seedvr2_vae_path.strip():
        return "SEEDVR2_VAE_PATH is required for SeedVR2 modes."
    return None
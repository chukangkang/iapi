import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlretrieve

from PIL import Image

from app.config import Settings


logger = logging.getLogger(__name__)

CODEFORMER_MODEL_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"


@dataclass(frozen=True)
class FaceCandidate:
    face_index: int
    bbox: tuple[float, float, float, float]
    detection_score: float
    original_face: Image.Image
    restored_face: Image.Image
    affine_matrix: tuple[tuple[float, ...], ...]
    landmarks: tuple[tuple[float, float], ...]
    identity_score: Optional[float] = None
    identity_accepted: bool = False
    original_aligned_landmarks: tuple[tuple[float, float], ...] = ()
    restored_aligned_landmarks: tuple[tuple[float, float], ...] = ()
    landmark_deformation_rms: Optional[float] = None
    landmark_deformation_max: Optional[float] = None
    landmark_accepted: bool = False
    quality_score: Optional[float] = None
    composite_score: Optional[float] = None
    selected: bool = False
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class FaceCandidateResult:
    source_image: Image.Image
    candidates: tuple[FaceCandidate, ...]
    detected_face_count: int = 0


class _SpandrelCodeFormer:
    def __init__(self, descriptor: Any, torch: Any, device: Any, use_half: bool):
        self.descriptor = descriptor
        self.torch = torch
        self.device = device
        self.use_half = use_half

    def restore(self, face: Image.Image, *, fidelity_weight: float) -> Image.Image:
        import numpy as np

        rgb = np.asarray(face.convert("RGB"), dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        tensor = tensor.mul(2).sub(1)
        if self.use_half:
            tensor = tensor.half()
        with self.torch.inference_mode():
            output = self.descriptor.model(tensor, weight=fidelity_weight)[0]
        output = output.squeeze(0).float().add(1).div(2).clamp_(0, 1).cpu().numpy()
        output = (output.transpose(1, 2, 0) * 255.0).round().astype("uint8")
        return Image.fromarray(output, mode="RGB")


class CodeFormerService:
    """Detect and restore aligned faces, returning candidates without pasting them back."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Optional[_SpandrelCodeFormer] = None
        self._model_key: Optional[tuple[Path, str, bool]] = None

    async def generate_candidates(self, image: Image.Image) -> FaceCandidateResult:
        if not self.settings.codeformer_enabled:
            return FaceCandidateResult(source_image=image, candidates=())
        try:
            return await asyncio.to_thread(self._generate_sync, image)
        except Exception as exc:
            logger.warning("CodeFormer candidate generation failed; keeping the source image: %s", exc)
            return FaceCandidateResult(source_image=image, candidates=())

    def resolve_model_path(self) -> Path:
        configured = self.settings.codeformer_model_path.strip()
        model_path = Path(configured) if configured else Path("weights/CodeFormer/codeformer.pth")
        if model_path.is_dir() or not model_path.suffix:
            model_path /= "codeformer.pth"
        if model_path.suffix.lower() != ".pth":
            raise ValueError(f"CodeFormer checkpoint must be a .pth file, got: {model_path}")
        return model_path

    def ensure_model_path(self) -> Path:
        model_path = self.resolve_model_path()
        if model_path.is_file():
            return model_path
        if not self.settings.codeformer_auto_download:
            raise FileNotFoundError(f"CodeFormer checkpoint does not exist: {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = model_path.with_suffix(".pth.part")
        logger.info("Downloading CodeFormer checkpoint from %s to %s", CODEFORMER_MODEL_URL, model_path)
        try:
            urlretrieve(CODEFORMER_MODEL_URL, temporary_path)
            temporary_path.replace(model_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return model_path

    def _generate_sync(self, image: Image.Image) -> FaceCandidateResult:
        device = self._resolve_device()
        model = self._get_model(device)
        helper = self._create_face_helper(device)
        source_rgb = image.convert("RGB")
        source_bgr = self._image_to_bgr(source_rgb)
        helper.clean_all()
        helper.read_image(source_bgr)
        detected_count = helper.get_face_landmarks_5(
            only_center_face=False,
            resize=640,
            eye_dist_threshold=5,
        )
        helper.align_warp_face()

        candidates = []
        landmarks = getattr(helper, "landmarks_5", ())
        matrices = getattr(helper, "affine_matrices", ())
        detections = getattr(helper, "det_faces", ())
        for index, cropped_face in enumerate(helper.cropped_faces):
            try:
                original_face = self._face_to_rgb_image(cropped_face)
                restored_face = model.restore(
                    original_face,
                    fidelity_weight=self.settings.codeformer_fidelity_weight,
                )
            except Exception as exc:
                logger.warning("CodeFormer failed for face %s; skipping candidate: %s", index, exc)
                continue
            detection = detections[index] if index < len(detections) else (0, 0, 0, 0, 0)
            matrix = matrices[index] if index < len(matrices) else ((1, 0, 0), (0, 1, 0))
            face_landmarks = landmarks[index] if index < len(landmarks) else ()
            candidates.append(
                FaceCandidate(
                    face_index=index,
                    bbox=tuple(float(value) for value in detection[:4]),
                    detection_score=float(detection[4]) if len(detection) > 4 else 0.0,
                    original_face=original_face,
                    restored_face=restored_face,
                    affine_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
                    landmarks=tuple(tuple(float(value) for value in point[:2]) for point in face_landmarks),
                )
            )
        return FaceCandidateResult(
            source_image=source_rgb,
            candidates=tuple(candidates),
            detected_face_count=int(detected_count or len(detections)),
        )

    @staticmethod
    def _face_to_rgb_image(face: Any) -> Image.Image:
        if isinstance(face, Image.Image):
            return face.convert("RGB")
        return Image.fromarray(face[:, :, ::-1], mode="RGB")

    @staticmethod
    def _image_to_bgr(image: Image.Image) -> Any:
        import numpy as np

        return np.asarray(image)[:, :, ::-1].copy()

    def _resolve_device(self) -> Any:
        import torch

        if self.settings.device == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    def _get_model(self, device: Any) -> _SpandrelCodeFormer:
        import torch
        import spandrel
        import spandrel_extra_arches

        model_path = self.ensure_model_path()
        use_half = device.type == "cuda" and self.settings.torch_dtype in {"float16", "bfloat16"}
        key = (model_path.resolve(), str(device), use_half)
        if self._model is not None and self._model_key == key:
            return self._model
        spandrel_extra_arches.install(ignore_duplicates=True)
        descriptor = spandrel.ModelLoader(device=device).load_from_file(model_path)
        if descriptor.architecture.id != "CodeFormer":
            raise ValueError(f"Expected a CodeFormer checkpoint, detected {descriptor.architecture.id}")
        descriptor.eval().to(device)
        if use_half and descriptor.supports_half:
            descriptor.half()
        else:
            use_half = False
        self._model = _SpandrelCodeFormer(descriptor, torch, device, use_half)
        self._model_key = (model_path.resolve(), str(device), use_half)
        return self._model

    @staticmethod
    def _create_face_helper(device: Any) -> Any:
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        return FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
        )
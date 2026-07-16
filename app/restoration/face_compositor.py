import logging
import math
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter

from app.config import Settings
from app.restoration.codeformer_service import FaceCandidate, FaceCandidateResult


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceCompositeResult:
    image: Image.Image
    pasted_face_count: int


class FaceSoftMaskCompositor:
    """Paste accepted aligned faces back with a feathered elliptical mask."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def composite(self, result: FaceCandidateResult) -> FaceCompositeResult:
        canvas = result.source_image.convert("RGB").copy()
        pasted = 0
        for candidate in result.candidates:
            if not candidate.selected:
                continue
            try:
                canvas = self._paste_candidate(canvas, candidate)
                pasted += 1
            except (TypeError, ValueError) as exc:
                logger.warning("Cannot paste face %s; invalid transform: %s", candidate.face_index, exc)
        return FaceCompositeResult(image=canvas, pasted_face_count=pasted)

    def _paste_candidate(self, canvas: Image.Image, candidate: FaceCandidate) -> Image.Image:
        coefficients = self._affine_coefficients(candidate)
        face = candidate.restored_face.convert("RGB")
        mask = self._create_soft_mask(face.size)
        output_size = canvas.size

        warped_face = face.transform(
            output_size,
            Image.Transform.AFFINE,
            coefficients,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0),
        )
        warped_mask = mask.transform(
            output_size,
            Image.Transform.AFFINE,
            coefficients,
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
        if self.settings.face_mask_opacity < 1.0:
            opacity = self.settings.face_mask_opacity
            warped_mask = warped_mask.point(lambda value: round(value * opacity))
        return Image.composite(warped_face, canvas, warped_mask)

    def _create_soft_mask(self, size: tuple[int, int]) -> Image.Image:
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("restored face has an empty size")
        shortest = min(width, height)
        inset = shortest * self.settings.face_mask_inset_ratio
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            (inset, inset, width - 1 - inset, height - 1 - inset),
            fill=255,
        )
        blur_radius = shortest * self.settings.face_mask_blur_ratio
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return mask

    @staticmethod
    def _affine_coefficients(candidate: FaceCandidate) -> tuple[float, float, float, float, float, float]:
        matrix = candidate.affine_matrix
        if len(matrix) != 2 or any(len(row) != 3 for row in matrix):
            raise ValueError("affine matrix must be 2x3")
        coefficients = tuple(float(value) for row in matrix for value in row)
        if not all(math.isfinite(value) for value in coefficients):
            raise ValueError("affine matrix contains a non-finite value")
        a, b, _, d, e, _ = coefficients
        if abs(a * e - b * d) <= 1e-8:
            raise ValueError("affine matrix is singular")
        return coefficients
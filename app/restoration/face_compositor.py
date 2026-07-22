import logging
import math
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageStat

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
        face = self._prepare_restored_face(candidate)
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

    def _prepare_restored_face(self, candidate: FaceCandidate) -> Image.Image:
        original = candidate.original_face.convert("RGB")
        restored = candidate.restored_face.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
        if self.settings.face_color_match_enabled and self.settings.face_color_match_strength > 0.0:
            restored = self._match_color(restored, original, self.settings.face_color_match_strength)
        texture_blend = self.settings.face_texture_blend
        if texture_blend > 0.0:
            restored = Image.blend(restored, original, texture_blend)
        return restored

    @staticmethod
    def _match_color(restored: Image.Image, original: Image.Image, strength: float) -> Image.Image:
        width, height = original.size
        sample_box = (
            max(0, round(width * 0.18)),
            max(0, round(height * 0.16)),
            min(width, round(width * 0.82)),
            min(height, round(height * 0.84)),
        )
        source_stats = ImageStat.Stat(original.crop(sample_box))
        restored_stats = ImageStat.Stat(restored.crop(sample_box))
        channels = []
        for restored_channel, source_mean, source_stddev, restored_mean, restored_stddev in zip(
            restored.split(),
            source_stats.mean,
            source_stats.stddev,
            restored_stats.mean,
            restored_stats.stddev,
        ):
            scale = source_stddev / max(restored_stddev, 1e-6)
            scale = max(0.75, min(1.25, scale))
            adjusted = ImageEnhance.Contrast(restored_channel).enhance(scale)
            adjusted_mean = ImageStat.Stat(adjusted).mean[0]
            offset = max(-32.0, min(32.0, source_mean - adjusted_mean))
            adjusted = adjusted.point(lambda value, delta=offset: max(0, min(255, round(value + delta))))
            channels.append(adjusted)
        matched = Image.merge("RGB", tuple(channels))
        return Image.blend(restored, matched, strength)

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
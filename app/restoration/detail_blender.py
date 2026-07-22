import logging
import threading

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.config import Settings


logger = logging.getLogger(__name__)
_face_detector = None
_face_detector_lock = threading.Lock()


def _get_face_detector():
    global _face_detector
    if _face_detector is not None:
        return _face_detector
    with _face_detector_lock:
        if _face_detector is None:
            from facexlib.detection import init_detection_model

            _face_detector = init_detection_model("retinaface_resnet50", half=False, device="cpu")
    return _face_detector


class RestorationDetailBlender:
    """Keep source tone/structure while borrowing plausible detail from a generative result."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def blend(self, source: Image.Image, restored: Image.Image) -> Image.Image:
        restored_rgb = restored.convert("RGB")
        if not self.settings.restoration_generative_blend_enabled:
            return restored_rgb

        source_rgb = source.convert("RGB").resize(restored_rgb.size, Image.Resampling.LANCZOS)
        strength = self.settings.restoration_generative_blend_strength
        if strength <= 0.0:
            return source_rgb

        radius = self.settings.restoration_generative_blend_low_frequency_radius
        if radius <= 0.0:
            candidate = Image.blend(source_rgb, restored_rgb, strength)
        else:
            # Reuse the generated high-frequency residual, but anchor luminance,
            # color and large shapes to the source. This suppresses hallucinated
            # low-frequency changes without throwing away recovered texture.
            low_frequency = restored_rgb.filter(ImageFilter.GaussianBlur(radius=radius))
            detail_only = ImageChops.subtract(restored_rgb, low_frequency, scale=1.0, offset=128)
            generated_detail = ImageChops.add(source_rgb, detail_only, scale=1.0, offset=-128)
            candidate = Image.blend(source_rgb, generated_detail, strength)

        face_mask = self._face_mask(source_rgb)
        if face_mask is None:
            return candidate
        skin_strength = min(strength, self.settings.restoration_generative_blend_skin_strength)
        face_candidate = Image.blend(source_rgb, generated_detail, skin_strength)
        return Image.composite(face_candidate, candidate, face_mask)

    @staticmethod
    def _face_mask(image: Image.Image) -> Image.Image | None:
        try:
            import numpy as np

            detector = _get_face_detector()
            detections = detector.detect_faces(np.asarray(image)[:, :, ::-1].copy(), 0.85)
        except Exception as exc:
            logger.debug("Face-aware generative blending unavailable: %s", exc)
            return None

        if detections is None or len(detections) == 0:
            return None
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for detection in detections:
            left, top, right, bottom = (float(value) for value in detection[:4])
            width = max(1.0, right - left)
            height = max(1.0, bottom - top)
            expand_x = width * 0.18
            expand_y = height * 0.22
            draw.ellipse(
                (
                    max(0.0, left - expand_x),
                    max(0.0, top - expand_y),
                    min(float(image.width - 1), right + expand_x),
                    min(float(image.height - 1), bottom + expand_y),
                ),
                fill=255,
            )
        blur_radius = max(2.0, min(image.size) * 0.012)
        return mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
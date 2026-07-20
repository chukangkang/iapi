from dataclasses import dataclass
import logging

from PIL import Image

from app.config import Settings
from app.restoration.analyzer import DegradationAnalyzer, DegradationReport
from app.restoration.style_classifier import get_style_classifier


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RestorationPlan:
    mode: str
    report: DegradationReport
    use_qwen_edit: bool
    upscale_method: str
    realesrgan_model_name: str
    face_restoration: bool
    use_swinir: bool
    use_supir: bool
    severe_blur: bool = False
    use_qwen_unblur_lora: bool = False
    is_illustration: bool = False
    illustration_score: float = 0.0
    style_label: str = "not_checked"


class RestorationOrchestrator:
    def __init__(self, settings: Settings, *, style_classifier=None):
        self.settings = settings
        self.analyzer = DegradationAnalyzer(anime_score_threshold=settings.restoration_anime_score_threshold)
        self.style_classifier = style_classifier or get_style_classifier(settings)

    def plan(self, requested_mode: str, image: Image.Image) -> RestorationPlan:
        report = self.analyzer.analyze(image)
        mode = report.recommended_mode if requested_mode == "auto" else requested_mode
        logger.info(
            "Restoration analysis: requested=%s recommended=%s blur=%.3f detail=%.3f noise=%.3f "
            "blockiness=%.3f anime_score=%.3f anime=%s threshold=%.3f",
            requested_mode,
            report.recommended_mode,
            report.blur_score,
            report.detail_score,
            report.noise_score,
            report.blockiness_score,
            report.anime_score,
            report.is_anime,
            self.settings.restoration_anime_score_threshold,
        )
        if mode not in {"preserve", "balanced", "creative"}:
            raise ValueError(f"Unsupported restoration mode: {mode}")

        illustration_score = 0.0
        style_label = "pixel_anime" if report.is_anime else "not_checked"
        semantic_illustration = False
        should_classify_style = (
            requested_mode == "auto"
            and self.settings.restoration_anime_detection_enabled
            and self.settings.restoration_style_classifier_enabled
            and not report.is_anime
        )
        if should_classify_style:
            try:
                classification = self.style_classifier.classify(image)
                illustration_score = classification.illustration_score
                style_label = classification.label
                semantic_illustration = (
                    illustration_score >= self.settings.restoration_style_classifier_threshold
                    and illustration_score > classification.photo_score
                )
                logger.info(
                    "Restoration semantic style: label=%s illustration_score=%.3f photo_score=%.3f "
                    "threshold=%.3f illustration=%s",
                    classification.label,
                    illustration_score,
                    classification.photo_score,
                    self.settings.restoration_style_classifier_threshold,
                    semantic_illustration,
                )
            except Exception as exc:
                style_label = "classifier_error"
                logger.warning(
                    "Restoration style classification failed; keeping conservative pixel-analysis route: %s",
                    exc,
                )

        is_illustration = report.is_anime or semantic_illustration
        if requested_mode == "auto" and self.settings.restoration_anime_detection_enabled and is_illustration:
            logger.info("Auto restoration selected anime model: %s", self.settings.restoration_anime_realesrgan_model_name)
            return RestorationPlan(
                mode=mode,
                report=report,
                use_qwen_edit=False,
                upscale_method="realesrgan",
                realesrgan_model_name=self.settings.restoration_anime_realesrgan_model_name,
                face_restoration=False,
                use_swinir=False,
                use_supir=False,
                is_illustration=True,
                illustration_score=illustration_score,
                style_label=style_label,
            )

        severe_blur = (
            requested_mode == "auto"
            and self.settings.restoration_severe_blur_enabled
            and report.recommended_mode == "balanced"
            and report.blur_score >= self.settings.restoration_severe_blur_threshold
        )
        if severe_blur:
            supir_available = self.settings.supir_enabled and bool(self.settings.supir_base_url.strip())
            use_supir = self.settings.restoration_severe_blur_prefer_supir and supir_available
            use_qwen_edit = not use_supir and self.settings.restoration_severe_blur_use_qwen_edit
            if use_supir or use_qwen_edit:
                route = "SUPIR" if use_supir else "Qwen Edit"
                logger.info(
                    "Auto restoration escalated severe blur to %s: blur_score=%.3f threshold=%.3f",
                    route,
                    report.blur_score,
                    self.settings.restoration_severe_blur_threshold,
                )
                return RestorationPlan(
                    mode=mode,
                    report=report,
                    use_qwen_edit=use_qwen_edit,
                    upscale_method="realesrgan",
                    realesrgan_model_name=self.settings.restoration_creative_realesrgan_model_name,
                    face_restoration=self.settings.codeformer_enabled,
                    use_swinir=False,
                    use_supir=use_supir,
                    severe_blur=True,
                    use_qwen_unblur_lora=use_qwen_edit and self.settings.qwen_unblur_upscale_lora_enabled,
                )

        if mode == "preserve":
            return RestorationPlan(
                mode=mode,
                report=report,
                use_qwen_edit=False,
                upscale_method="realesrgan",
                realesrgan_model_name=self.settings.restoration_preserve_realesrgan_model_name,
                face_restoration=False,
                use_swinir=False,
                use_supir=False,
            )
        if mode == "balanced":
            use_swinir = self.settings.swinir_enabled and (
                requested_mode != "auto"
                or report.noise_score >= 0.20
                or report.blockiness_score >= 0.25
            )
            return RestorationPlan(
                mode=mode,
                report=report,
                use_qwen_edit=False,
                upscale_method="realesrgan",
                realesrgan_model_name=self.settings.restoration_balanced_realesrgan_model_name,
                face_restoration=self.settings.codeformer_enabled,
                use_swinir=use_swinir,
                use_supir=False,
            )
        return RestorationPlan(
            mode=mode,
            report=report,
            use_qwen_edit=True,
            upscale_method="realesrgan",
            realesrgan_model_name=self.settings.restoration_creative_realesrgan_model_name,
            face_restoration=self.settings.codeformer_enabled,
            use_swinir=False,
            use_supir=self.settings.supir_enabled and bool(self.settings.supir_base_url.strip()),
        )

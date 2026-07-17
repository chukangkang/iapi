from dataclasses import dataclass
import logging

from PIL import Image

from app.config import Settings
from app.restoration.analyzer import DegradationAnalyzer, DegradationReport


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


class RestorationOrchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.analyzer = DegradationAnalyzer(anime_score_threshold=settings.restoration_anime_score_threshold)

    def plan(self, requested_mode: str, image: Image.Image) -> RestorationPlan:
        report = self.analyzer.analyze(image)
        mode = report.recommended_mode if requested_mode == "auto" else requested_mode
        logger.info(
            "Restoration analysis: requested=%s recommended=%s anime_score=%.3f anime=%s threshold=%.3f",
            requested_mode,
            report.recommended_mode,
            report.anime_score,
            report.is_anime,
            self.settings.restoration_anime_score_threshold,
        )
        if mode not in {"preserve", "balanced", "creative"}:
            raise ValueError(f"Unsupported restoration mode: {mode}")

        if requested_mode == "auto" and self.settings.restoration_anime_detection_enabled and report.is_anime:
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
            )

        severe_blur = (
            requested_mode == "auto"
            and self.settings.restoration_severe_blur_enabled
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
            return RestorationPlan(
                mode=mode,
                report=report,
                use_qwen_edit=False,
                upscale_method="realesrgan",
                realesrgan_model_name=self.settings.restoration_balanced_realesrgan_model_name,
                face_restoration=self.settings.codeformer_enabled,
                use_swinir=self.settings.swinir_enabled,
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

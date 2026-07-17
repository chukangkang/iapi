from dataclasses import dataclass

from PIL import Image

from app.config import Settings
from app.restoration.analyzer import DegradationAnalyzer, DegradationReport


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


class RestorationOrchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.analyzer = DegradationAnalyzer(anime_score_threshold=settings.restoration_anime_score_threshold)

    def plan(self, requested_mode: str, image: Image.Image) -> RestorationPlan:
        report = self.analyzer.analyze(image)
        mode = report.recommended_mode if requested_mode == "auto" else requested_mode
        if mode not in {"preserve", "balanced", "creative"}:
            raise ValueError(f"Unsupported restoration mode: {mode}")

        if requested_mode == "auto" and self.settings.restoration_anime_detection_enabled and report.is_anime:
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

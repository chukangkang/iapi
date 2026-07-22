import logging
from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image

from app.config import Settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IllustrationClassification:
    illustration_score: float
    photo_score: float
    label: str


class IllustrationStyleClassifier:
    """Lazy CLIP classifier for photographic versus illustrated image content."""

    # Keep the two labels short, symmetric, and mutually exclusive. The old
    # four-label prompt compared detailed photography subtypes with a single
    # label containing many popular art concepts. CLIP strongly preferred that
    # label for soft studio portraits, routing real photos to the anime model.
    PHOTO_LABEL = "a real photograph"
    ILLUSTRATION_LABEL = "a non-photographic illustration or render"
    PHOTO_LABELS = (PHOTO_LABEL,)
    ILLUSTRATION_LABELS = (ILLUSTRATION_LABEL,)
    CANDIDATE_LABELS = (PHOTO_LABEL, ILLUSTRATION_LABEL)

    def __init__(self, settings: Settings):
        self.settings = settings
        self._classifier: Optional[Any] = None

    def classify(self, image: Image.Image) -> IllustrationClassification:
        classifier = self._get_classifier()
        predictions = classifier(
            image.convert("RGB"),
            candidate_labels=list(self.CANDIDATE_LABELS),
            hypothesis_template="{}",
        )
        scores = {
            str(prediction["label"]): float(prediction["score"])
            for prediction in predictions
        }
        photo_score = sum(scores.get(label, 0.0) for label in self.PHOTO_LABELS)
        illustration_score = sum(scores.get(label, 0.0) for label in self.ILLUSTRATION_LABELS)
        label = max(scores, key=scores.get) if scores else "unknown"
        return IllustrationClassification(
            illustration_score=max(0.0, min(1.0, illustration_score)),
            photo_score=max(0.0, min(1.0, photo_score)),
            label=label,
        )

    def _get_classifier(self):
        if self._classifier is not None:
            return self._classifier

        from transformers import AutoImageProcessor, pipeline

        pipeline_kwargs = {}
        if self.settings.hf_token and not self.settings.hf_token.startswith("replace-with"):
            pipeline_kwargs["token"] = self.settings.hf_token
        logger.info(
            "Loading restoration style classifier on CPU: model=%s",
            self.settings.restoration_style_classifier_model_path,
        )
        image_processor = AutoImageProcessor.from_pretrained(
            self.settings.restoration_style_classifier_model_path,
            use_fast=False,
            **pipeline_kwargs,
        )
        self._classifier = pipeline(
            task="zero-shot-image-classification",
            model=self.settings.restoration_style_classifier_model_path,
            image_processor=image_processor,
            device=-1,
            **pipeline_kwargs,
        )
        return self._classifier


_style_classifiers: dict[tuple[str, str], IllustrationStyleClassifier] = {}


def get_style_classifier(settings: Settings) -> IllustrationStyleClassifier:
    token = settings.hf_token if settings.hf_token and not settings.hf_token.startswith("replace-with") else ""
    key = (settings.restoration_style_classifier_model_path, token)
    classifier = _style_classifiers.get(key)
    if classifier is None:
        classifier = IllustrationStyleClassifier(settings)
        _style_classifiers[key] = classifier
    return classifier

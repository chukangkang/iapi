from dataclasses import asdict, dataclass

from PIL import Image, ImageChops, ImageFilter, ImageStat


@dataclass(frozen=True)
class DegradationReport:
    blur_score: float
    detail_score: float
    noise_score: float
    blockiness_score: float
    exposure_score: float
    recommended_mode: str
    anime_score: float = 0.0
    is_anime: bool = False

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


class DegradationAnalyzer:
    """Lightweight no-reference analysis used to select a restoration route."""

    def __init__(self, *, anime_score_threshold: float = 0.58):
        self.anime_score_threshold = anime_score_threshold

    def analyze(self, image: Image.Image) -> DegradationReport:
        color_sample = image.convert("RGB")
        color_sample.thumbnail((512, 512), Image.Resampling.BILINEAR)
        sample = image.convert("L")
        sample.thumbnail((512, 512), Image.Resampling.BILINEAR)
        edges = sample.filter(ImageFilter.FIND_EDGES)
        edge_mean = ImageStat.Stat(edges).mean[0] / 255.0
        detail_score = max(0.0, min(1.0, edge_mean * 4.0))
        blur_score = 1.0 - detail_score

        denoised = sample.filter(ImageFilter.MedianFilter(size=3))
        difference = self._mean_absolute_difference(sample, denoised) / 255.0
        noise_score = max(0.0, min(1.0, difference * 8.0))
        blockiness_score = self._blockiness(sample)
        exposure_score = self._exposure(sample)
        anime_score = self._anime_score(color_sample)

        if detail_score >= 0.6 and blur_score <= 0.4:
            recommended_mode = "preserve"
        elif blur_score >= 0.65 or noise_score >= 0.45 or blockiness_score >= 0.35:
            recommended_mode = "balanced"
        else:
            recommended_mode = "preserve"
        return DegradationReport(
            blur_score=blur_score,
            detail_score=detail_score,
            noise_score=noise_score,
            blockiness_score=blockiness_score,
            exposure_score=exposure_score,
            recommended_mode=recommended_mode,
            anime_score=anime_score,
            is_anime=anime_score >= self.anime_score_threshold,
        )

    @classmethod
    def _anime_score(cls, image: Image.Image) -> float:
        """Estimate flat-color illustration/anime style without loading another model."""
        quantized = image.quantize(colors=32, method=Image.Quantize.MEDIANCUT).convert("RGB")
        palette_error = cls._rgb_mean_absolute_difference(image, quantized) / 255.0
        palette_score = max(0.0, min(1.0, 1.0 - palette_error * 14.0))

        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_values = list(edges.get_flattened_data())
        strong_edge_ratio = sum(value >= 48 for value in edge_values) / max(1, len(edge_values))
        line_score = max(0.0, min(1.0, strong_edge_ratio / 0.12))

        saturation = ImageStat.Stat(image.convert("HSV").getchannel("S")).mean[0] / 255.0
        saturation_score = max(0.0, min(1.0, saturation / 0.35))
        return max(0.0, min(1.0, palette_score * 0.65 + line_score * 0.20 + saturation_score * 0.15))

    @staticmethod
    def _rgb_mean_absolute_difference(left: Image.Image, right: Image.Image) -> float:
        channel_means = ImageStat.Stat(ImageChops.difference(left, right)).mean
        return sum(channel_means) / max(1, len(channel_means))

    @staticmethod
    def _mean_absolute_difference(left: Image.Image, right: Image.Image) -> float:
        left_values = list(left.get_flattened_data())
        right_values = list(right.get_flattened_data())
        if not left_values:
            return 0.0
        return sum(abs(a - b) for a, b in zip(left_values, right_values)) / len(left_values)

    @staticmethod
    def _blockiness(image: Image.Image) -> float:
        pixels = image.load()
        boundary = []
        interior = []
        for y in range(image.height):
            for x in range(1, image.width):
                value = abs(pixels[x, y] - pixels[x - 1, y])
                (boundary if x % 8 == 0 else interior).append(value)
        if not boundary:
            return 0.0
        boundary_mean = sum(boundary) / len(boundary)
        interior_mean = sum(interior) / len(interior) if interior else 0.0
        return max(0.0, min(1.0, (boundary_mean - interior_mean) / 64.0))

    @staticmethod
    def _exposure(image: Image.Image) -> float:
        histogram = image.histogram()
        pixels = max(1, image.width * image.height)
        clipped = sum(histogram[:5]) + sum(histogram[-5:])
        return min(1.0, clipped / pixels)

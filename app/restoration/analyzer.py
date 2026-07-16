from dataclasses import asdict, dataclass

from PIL import Image, ImageFilter, ImageStat


@dataclass(frozen=True)
class DegradationReport:
    blur_score: float
    detail_score: float
    noise_score: float
    blockiness_score: float
    exposure_score: float
    recommended_mode: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


class DegradationAnalyzer:
    """Lightweight no-reference analysis used to select a restoration route."""

    def analyze(self, image: Image.Image) -> DegradationReport:
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
        )

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

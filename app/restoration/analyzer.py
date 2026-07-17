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

    def __init__(self, *, anime_score_threshold: float = 0.72):
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
        """Estimate line-art/flat-color style without loading a classifier model."""
        quantized = image.quantize(colors=16, method=Image.Quantize.MEDIANCUT).convert("RGB")
        palette_error = cls._rgb_mean_absolute_difference(image, quantized) / 255.0
        palette_score = cls._clamp_score(1.0 - palette_error * 22.0)

        gray = image.convert("L")
        raw_edge_histogram, raw_edge_pixels = cls._interior_histogram(gray.filter(ImageFilter.FIND_EDGES))
        smoothed = gray.filter(ImageFilter.GaussianBlur(radius=0.8))
        smooth_edge_histogram, smooth_edge_pixels = cls._interior_histogram(
            smoothed.filter(ImageFilter.FIND_EDGES)
        )
        raw_edge_ratio = sum(raw_edge_histogram[48:]) / max(1, raw_edge_pixels)
        smooth_edge_ratio = sum(smooth_edge_histogram[32:]) / max(1, smooth_edge_pixels)
        edge_persistence = smooth_edge_ratio / max(raw_edge_ratio, 1e-6)
        line_score = min(
            cls._clamp_score(raw_edge_ratio / 0.06),
            cls._clamp_score(smooth_edge_ratio / 0.04),
            cls._clamp_score(edge_persistence / 0.65),
        )

        difference_histogram, difference_pixels = cls._neighbor_difference_histogram(gray)
        flat_pixel_ratio = sum(difference_histogram[:3]) / max(1, difference_pixels)
        flat_color_score = cls._clamp_score((flat_pixel_ratio - 0.55) / 0.30)

        smooth_difference_histogram, smooth_difference_pixels = cls._neighbor_difference_histogram(smoothed)
        continuous_tone_ratio = sum(smooth_difference_histogram[2:13]) / max(1, smooth_difference_pixels)
        flat_tone_score = cls._clamp_score(1.0 - (continuous_tone_ratio - 0.10) / 0.30)

        saturation = ImageStat.Stat(image.convert("HSV").getchannel("S")).mean[0] / 255.0
        saturation_score = cls._clamp_score(saturation / 0.30)

        base_score = (
            palette_score * 0.30
            + line_score * 0.25
            + saturation_score * 0.10
            + flat_color_score * 0.20
            + flat_tone_score * 0.15
        )
        # Photographs can have a compact palette and many strong edges. Requiring
        # edges that survive light blur plus large flat regions prevents textured
        # scenes and real faces from crossing the anime threshold.
        structure_gate = 0.65 + line_score * 0.35
        tone_gate = 0.65 + flat_tone_score * 0.35
        return cls._clamp_score(base_score * structure_gate * tone_gate)

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _interior_histogram(image: Image.Image) -> tuple[list[int], int]:
        if image.width > 2 and image.height > 2:
            image = image.crop((1, 1, image.width - 1, image.height - 1))
        return image.histogram(), image.width * image.height

    @staticmethod
    def _neighbor_difference_histogram(image: Image.Image) -> tuple[list[int], int]:
        histogram = [0] * 256
        pixel_count = 0
        if image.width > 1:
            horizontal = ImageChops.difference(
                image.crop((1, 0, image.width, image.height)),
                image.crop((0, 0, image.width - 1, image.height)),
            )
            histogram = [left + right for left, right in zip(histogram, horizontal.histogram())]
            pixel_count += horizontal.width * horizontal.height
        if image.height > 1:
            vertical = ImageChops.difference(
                image.crop((0, 1, image.width, image.height)),
                image.crop((0, 0, image.width, image.height - 1)),
            )
            histogram = [left + right for left, right in zip(histogram, vertical.histogram())]
            pixel_count += vertical.width * vertical.height
        return histogram, pixel_count

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

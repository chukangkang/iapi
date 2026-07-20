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
        detail_score = self._edge_detail_score(sample)
        blur_score = 1.0 - detail_score

        denoised = sample.filter(ImageFilter.MedianFilter(size=3))
        difference = self._mean_absolute_difference(sample, denoised) / 255.0
        directional_balance = self._directional_activity_balance(sample, threshold=8)
        noise_score = self._clamp_score(difference * 8.0 * directional_balance)
        blockiness_score = self._blockiness(sample)
        exposure_score = self._exposure(sample)
        anime_score = self._anime_score(color_sample)

        if noise_score >= 0.45 or blockiness_score >= 0.35:
            recommended_mode = "balanced"
        elif blur_score >= 0.65:
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
    def _edge_detail_score(cls, image: Image.Image) -> float:
        """Measure edge sharpness independently of how much texture an image contains."""
        edge_histogram, edge_pixels = cls._neighbor_difference_histogram(image)
        edge_strength = cls._top_histogram_mean(edge_histogram, edge_pixels, fraction=0.01)
        if edge_strength <= 0.0:
            return 0.0

        smoothed = image.filter(ImageFilter.GaussianBlur(radius=0.8))
        smooth_histogram, smooth_pixels = cls._neighbor_difference_histogram(smoothed)
        smooth_strength = cls._top_histogram_mean(smooth_histogram, smooth_pixels, fraction=0.01)
        edge_decay = edge_strength / max(smooth_strength, 1e-6)

        # Crisp edges lose substantially more contrast under a light blur than
        # already-blurred edges. Looking only at the strongest one percent keeps
        # smooth skin/background areas from making a clear portrait look blurry.
        sharpness_score = cls._clamp_score((edge_decay - 1.04) / 0.30)
        edge_evidence = cls._clamp_score(edge_strength / 8.0)
        return sharpness_score * edge_evidence

    @staticmethod
    def _top_histogram_mean(
        histogram: list[int],
        pixel_count: int,
        *,
        fraction: float,
    ) -> float:
        target_count = max(1, round(pixel_count * fraction))
        selected_count = 0
        weighted_sum = 0
        for value in range(len(histogram) - 1, -1, -1):
            count = min(histogram[value], target_count - selected_count)
            weighted_sum += value * count
            selected_count += count
            if selected_count >= target_count:
                break
        return weighted_sum / max(1, selected_count)

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
    def _directional_activity_balance(image: Image.Image, *, threshold: int) -> float:
        if image.width <= 1 or image.height <= 1:
            return 0.0
        horizontal = ImageChops.difference(
            image.crop((1, 0, image.width, image.height)),
            image.crop((0, 0, image.width - 1, image.height)),
        )
        vertical = ImageChops.difference(
            image.crop((0, 1, image.width, image.height)),
            image.crop((0, 0, image.width, image.height - 1)),
        )
        horizontal_histogram = horizontal.histogram()
        vertical_histogram = vertical.histogram()
        horizontal_ratio = sum(horizontal_histogram[threshold:]) / max(1, horizontal.width * horizontal.height)
        vertical_ratio = sum(vertical_histogram[threshold:]) / max(1, vertical.width * vertical.height)
        return min(horizontal_ratio, vertical_ratio) / max(horizontal_ratio, vertical_ratio, 1e-6)

    @classmethod
    def _blockiness(cls, image: Image.Image) -> float:
        horizontal_score = cls._blockiness_axis(image, axis="horizontal")
        vertical_score = cls._blockiness_axis(image, axis="vertical")
        # JPEG blocks form a two-dimensional 8x8 grid. A strong edge pattern in
        # only one direction is usually text, architecture, or line art.
        return min(horizontal_score, vertical_score)

    @staticmethod
    def _blockiness_axis(image: Image.Image, *, axis: str) -> float:
        pixels = image.load()
        boundary = []
        interior = []
        if axis == "horizontal":
            for y in range(image.height):
                for x in range(1, image.width):
                    value = abs(pixels[x, y] - pixels[x - 1, y])
                    (boundary if x % 8 == 0 else interior).append(value)
        else:
            for y in range(1, image.height):
                for x in range(image.width):
                    value = abs(pixels[x, y] - pixels[x, y - 1])
                    (boundary if y % 8 == 0 else interior).append(value)
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

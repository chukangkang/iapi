import argparse
import base64
import csv
import io
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageChops, ImageFilter, ImageStat


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def summarize_records(records: list[dict]) -> dict:
    successful = [record for record in records if record["ok"]]
    latencies = sorted(float(record["elapsed_ms"]) for record in successful)
    elapsed_seconds = sum(latencies) / 1000.0
    megapixels = sum(float(record["output_megapixels"]) for record in successful)
    return {
        "total": len(records),
        "succeeded": len(successful),
        "failed": len(records) - len(successful),
        "success_rate": len(successful) / len(records) if records else 0.0,
        "latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "throughput_megapixels_per_second": megapixels / elapsed_seconds if elapsed_seconds else 0.0,
    }


def _percentile(values: list[float], quantile: float):
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def benchmark(args) -> tuple[list[dict], dict]:
    image_paths = sorted(path for path in args.input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No real benchmark images found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for image_path in image_paths:
        record = {"input": str(image_path), "ok": False, "elapsed_ms": 0.0, "output_megapixels": 0.0}
        started = time.perf_counter()
        try:
            source = Image.open(image_path).convert("RGB")
            width = args.width or source.width * args.upscale
            height = args.height or source.height * args.upscale
            restored = _request_restore(args, source, width, height)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            output_path = args.output_dir / f"{image_path.stem}_supir.png"
            restored.save(output_path)
            reference = source.resize(restored.size, Image.Resampling.LANCZOS)
            record.update(
                ok=True,
                output=str(output_path),
                elapsed_ms=elapsed_ms,
                output_megapixels=restored.width * restored.height / 1_000_000.0,
                detail_gain=_detail(restored) - _detail(reference),
                fidelity_mae=_mae(reference, restored),
            )
        except Exception as exc:
            record.update(elapsed_ms=(time.perf_counter() - started) * 1000.0, error=str(exc))
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
    return records, summarize_records(records)


def _request_restore(args, image: Image.Image, width: int, height: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = json.dumps(
        {
            "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "prompt": args.prompt,
            "width": width,
            "height": height,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = Request(f"{args.base_url.rstrip('/')}/v1/restore", data=payload, headers=headers, method="POST")
    with urlopen(request, timeout=args.timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return Image.open(io.BytesIO(base64.b64decode(result["image"]))).convert("RGB")


def _detail(image: Image.Image) -> float:
    return ImageStat.Stat(image.convert("L").filter(ImageFilter.FIND_EDGES)).mean[0] / 255.0


def _mae(left: Image.Image, right: Image.Image) -> float:
    return ImageStat.Stat(ImageChops.difference(left, right)).mean[0] / 255.0


def _write_reports(output_dir: Path, records: list[dict], summary: dict) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({key for record in records for key in record})
    with (output_dir / "records.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the standalone SUPIR Worker with real images")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/supir"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--prompt", default="Restore this photograph naturally while preserving identity and composition.")
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    records, summary = benchmark(args)
    _write_reports(args.output_dir, records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
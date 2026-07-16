import pytest

from scripts.benchmark_supir import summarize_records


def test_benchmark_summary_computes_latency_percentiles_and_success_rate():
    summary = summarize_records(
        [
            {"ok": True, "elapsed_ms": 100.0, "output_megapixels": 1.0},
            {"ok": True, "elapsed_ms": 300.0, "output_megapixels": 2.0},
            {"ok": False, "elapsed_ms": 50.0, "output_megapixels": 0.0},
        ]
    )

    assert summary["total"] == 3
    assert summary["succeeded"] == 2
    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["latency_ms_mean"] == pytest.approx(200.0)
    assert summary["latency_ms_p50"] == pytest.approx(200.0)
    assert summary["throughput_megapixels_per_second"] == pytest.approx(7.5)
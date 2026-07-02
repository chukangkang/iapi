from app.config import Settings
from app.pipeline_utils import apply_pipeline_cpu_offload


class FakePipeline:
    def __init__(self):
        self.calls = []

    def enable_model_cpu_offload(self):
        self.calls.append("model")

    def enable_sequential_cpu_offload(self):
        self.calls.append("sequential")


def test_cpu_offload_defaults_to_model_offload():
    pipe = FakePipeline()
    settings = Settings(enable_cpu_offload=True)

    assert apply_pipeline_cpu_offload(pipe, settings, "cuda") is True
    assert pipe.calls == ["model"]


def test_cpu_offload_can_use_sequential_mode():
    pipe = FakePipeline()
    settings = Settings(enable_cpu_offload=True, cpu_offload_mode="sequential")

    assert apply_pipeline_cpu_offload(pipe, settings, "cuda") is True
    assert pipe.calls == ["sequential"]


def test_cpu_offload_skips_non_cuda_devices():
    pipe = FakePipeline()
    settings = Settings(enable_cpu_offload=True)

    assert apply_pipeline_cpu_offload(pipe, settings, "cpu") is False
    assert pipe.calls == []
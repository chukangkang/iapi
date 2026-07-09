from app.config import Settings
from app.pipeline_utils import apply_pipeline_cpu_offload, get_pipeline_device_map_kwargs


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


class FakeCudaDeviceProperties:
    total_memory = 24 * 1024**3


class FakeCuda:
    @staticmethod
    def device_count():
        return 4

    @staticmethod
    def get_device_properties(index):
        return FakeCudaDeviceProperties()


class FakeTorch:
    cuda = FakeCuda()


def test_device_map_skips_single_gpu_config():
    settings = Settings(model_gpu_count=1)

    assert get_pipeline_device_map_kwargs(settings, FakeTorch, "cuda") == {}


def test_device_map_uses_balanced_for_multi_gpu_config():
    settings = Settings(model_gpu_count=4)

    kwargs = get_pipeline_device_map_kwargs(settings, FakeTorch, "cuda")

    assert kwargs == {
        "device_map": "balanced",
        "max_memory": {0: "24GiB", 1: "24GiB", 2: "24GiB", 3: "24GiB"},
    }


def test_device_map_applies_explicit_memory_limit():
    settings = Settings(model_gpu_count=2, model_gpu_memory_limit="20GiB")

    kwargs = get_pipeline_device_map_kwargs(settings, FakeTorch, "cuda")

    assert kwargs == {
        "device_map": "balanced",
        "max_memory": {0: "20GiB", 1: "20GiB"},
    }
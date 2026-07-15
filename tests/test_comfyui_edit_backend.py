import pytest
from PIL import Image

from app.comfyui_edit_backend import ComfyUIEditBackend, _comfyui_import_error
from app.config import Settings


class FakeTensor:
    def __init__(self, value=None):
        self.value = value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def clamp(self, minimum, maximum):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, index):
        return self


class FakeTorch:
    float32 = "float32"
    float8_e4m3fn = "fp8_e4m3fn"
    float8_e5m2 = "fp8_e5m2"

    @staticmethod
    def from_numpy(value):
        return FakeTensor(value)


class NodeOutput:
    def __init__(self, *args):
        self.args = args

    def __getitem__(self, index):
        return self.args[index]


class FakeFolderPaths:
    def __init__(self):
        self.paths = []

    def add_model_folder_path(self, category, path, is_default=False):
        self.paths.append((category, path, is_default))

    @staticmethod
    def get_full_path_or_raise(category, name):
        return f"/{category}/{name}"

    @staticmethod
    def get_folder_paths(category):
        return [f"/{category}"]


class FakeComfySD:
    class CLIPType:
        QWEN_IMAGE = "qwen_image"

    def __init__(self, calls):
        self.calls = calls

    def load_diffusion_model(self, path, model_options):
        self.calls.append(("unet", path, model_options))
        return "model"

    def load_clip(self, **kwargs):
        self.calls.append(("clip", kwargs))
        return "clip"

    def VAE(self, *, sd, metadata):
        self.calls.append(("vae", sd, metadata))
        return "vae"

    def load_lora_for_models(self, model, clip, lora, strength_model, strength_clip, **kwargs):
        self.calls.append(("lora", model, strength_model))
        return ("lora-model", clip)


class FakeComfyUtils:
    def __init__(self, calls):
        self.calls = calls

    def load_torch_file(self, path, **kwargs):
        self.calls.append(("load_file", path, kwargs))
        if kwargs.get("return_metadata"):
            return ({"weights": path}, {"meta": True})
        return {"weights": path}


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.folder_paths = FakeFolderPaths()
        self.torch = FakeTorch
        self.sd = FakeComfySD(self.calls)
        self.utils = FakeComfyUtils(self.calls)

    def patch_aura(self, model, shift):
        self.calls.append(("sampling", model, shift))
        return "patched-model"

    def encode_edit(self, clip, prompt, vae, image1, image2=None, image3=None):
        self.calls.append(("positive", clip, prompt, vae, image1))
        return "positive"

    def encode_negative(self, clip, text):
        self.calls.append(("negative", clip, text))
        return ("negative",)

    def empty_latent(self, width, height, batch_size):
        self.calls.append(("latent", width, height, batch_size))
        return ({"samples": "latent"},)

    def sample(self, **kwargs):
        self.calls.append(("sample", kwargs))
        return ({"samples": "sampled"},)

    def decode(self, vae, samples):
        self.calls.append(("decode", vae, samples))
        return (FakeTensor([[[[1.0, 0.0, 0.0]]]]),)

    def cleanup(self):
        self.calls.append(("cleanup",))


def make_settings(tmp_path, **overrides):
    comfy_path = tmp_path / "ComfyUI"
    models_path = comfy_path / "models"
    comfy_path.mkdir()
    models_path.mkdir()
    defaults = {
        "qwen_edit_backend": "comfyui",
        "comfyui_path": comfy_path,
        "comfyui_models_path": models_path,
        "comfyui_qwen_edit_unet_name": "qwen_image_edit_2511_fp8mixed.safetensors",
        "comfyui_qwen_edit_clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "comfyui_qwen_edit_vae_name": "qwen_image_vae.safetensors",
        "comfyui_qwen_edit_lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_backend_runs_comfy_qwen_edit_node_chain(tmp_path, monkeypatch):
    runtime = FakeRuntime()
    backend = ComfyUIEditBackend(make_settings(tmp_path), runtime=runtime)
    monkeypatch.setattr(backend, "_pil_to_tensor", lambda image: "image-tensor")
    monkeypatch.setattr(backend, "_tensor_to_pil", lambda tensor: Image.new("RGB", (1, 1), "red"))

    output = backend.edit(
        prompt="restore",
        negative_prompt="bad",
        image=Image.new("RGB", (16, 16)),
        width=512,
        height=512,
        num_inference_steps=4,
        seed=7,
        guidance_scale=1.0,
        strength=1.0,
    )

    assert output.size == (1, 1)
    assert ("sampling", "lora-model", pytest.approx(3.1)) in runtime.calls
    sample_call = next(call for call in runtime.calls if call[0] == "sample")
    assert sample_call[1]["model"] == "patched-model"
    assert sample_call[1]["positive"] == "positive"
    assert sample_call[1]["negative"] == "negative"
    assert sample_call[1]["sampler_name"] == "euler"
    assert sample_call[1]["scheduler"] == "simple"
    assert sample_call[1]["denoise"] == 1.0


def test_components_are_cached_between_edits(tmp_path, monkeypatch):
    runtime = FakeRuntime()
    backend = ComfyUIEditBackend(make_settings(tmp_path), runtime=runtime)
    monkeypatch.setattr(backend, "_pil_to_tensor", lambda image: "image-tensor")
    monkeypatch.setattr(backend, "_tensor_to_pil", lambda tensor: Image.new("RGB", (1, 1)))

    kwargs = dict(prompt="x", negative_prompt=None, image=Image.new("RGB", (8, 8)), width=512, height=512,
                  num_inference_steps=4, seed=1, guidance_scale=1.0, strength=1.0)
    backend.edit(**kwargs)
    backend.edit(**kwargs)

    assert len([call for call in runtime.calls if call[0] == "unet"]) == 1
    assert len([call for call in runtime.calls if call[0] == "clip"]) == 1
    assert len([call for call in runtime.calls if call[0] == "vae"]) == 1
    assert len([call for call in runtime.calls if call[0] == "lora"]) == 1


def test_missing_comfyui_source_has_actionable_error(tmp_path):
    settings = Settings(qwen_edit_backend="comfyui", comfyui_path=tmp_path / "missing")

    with pytest.raises(RuntimeError, match="COMFYUI_PATH"):
        ComfyUIEditBackend(settings).prepare()


def test_missing_transitive_dependency_has_install_command():
    message = _comfyui_import_error(ModuleNotFoundError("No module named 'trampoline'"))

    assert "trampoline" in message
    assert "pip install" in message
    assert "requirements-worker.txt" in message



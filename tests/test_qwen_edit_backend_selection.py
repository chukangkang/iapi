from PIL import Image

from app.config import Settings
from app.qwen_edit_service import QwenImageEditService


class FakeComfyBackend:
    def __init__(self):
        self.prepare_calls = []
        self.edit_calls = []
        self.unloaded = False

    def prepare(self, **kwargs):
        self.prepare_calls.append(kwargs)

    def edit(self, **kwargs):
        self.edit_calls.append(kwargs)
        return Image.new("RGB", (2, 2), "blue")

    def unload(self):
        self.unloaded = True


def test_edit_dispatches_to_comfyui_backend(monkeypatch):
    backend = FakeComfyBackend()
    service = QwenImageEditService(Settings(qwen_edit_backend="comfyui"))
    monkeypatch.setattr(service, "_get_comfyui_backend", lambda: backend)

    output = service._edit_sync(
        prompt="edit",
        negative_prompt=None,
        image=Image.new("RGB", (8, 8)),
        width=512,
        height=512,
        num_inference_steps=4,
        seed=1,
        guidance_scale=1.0,
        strength=1.0,
        lora_path=None,
        lora_weight_name=None,
        lora_scale=1.0,
    )

    assert output.size == (2, 2)
    assert backend.edit_calls[0]["prompt"] == "edit"


def test_prepare_and_unload_dispatch_to_comfyui_backend(monkeypatch):
    backend = FakeComfyBackend()
    service = QwenImageEditService(Settings(qwen_edit_backend="comfyui"))
    service._comfyui_backend = backend

    service._prepare_sync(lora_path="repo", lora_weight_name="lora.safetensors", lora_scale=0.5)
    service.unload()

    assert backend.prepare_calls == [{
        "lora_path": "repo",
        "lora_weight_name": "lora.safetensors",
        "lora_scale": 0.5,
    }]
    assert backend.unloaded is True
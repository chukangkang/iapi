import pytest
import diffusers
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from app.image_utils import image_to_base64_png, string_list_to_images, string_to_image
from app.main import ImageGenerationRequest, _apply_prompt_params, _payload_image_to_reference, _resolve_dimensions, _resolve_enhance_mode, _resolve_face_enhance, _resolve_negative_prompt, _resolve_qwen_edit_dimensions, _resolve_qwen_unblur_lora, _validate_image_payload
from app.config import Settings
from app.qwen_edit_service import QwenImageEditService
from app.qwen_image_service import ModelManager
from app.upscale_service import ImageUpscaleService


def _sample_base64_png() -> str:
    image = Image.new("RGB", (2, 2), "red")
    return image_to_base64_png(image)


def test_string_to_image_accepts_raw_base64_with_whitespace():
    encoded = _sample_base64_png()
    wrapped = f"{encoded[:12]}\n {encoded[12:]}"

    image = string_to_image(wrapped)

    assert image is not None
    assert image.size == (2, 2)


def test_string_to_image_accepts_data_url_with_whitespace():
    encoded = _sample_base64_png()
    data_url = f"data:image/png;base64,{encoded[:10]}\n{encoded[10:]}"

    image = string_to_image(data_url)

    assert image is not None
    assert image.size == (2, 2)


def test_string_list_to_images_accepts_two_images():
    encoded = _sample_base64_png()

    images = string_list_to_images([encoded, encoded])

    assert len(images) == 2
    assert images[0].size == (2, 2)


def test_response_format_accepts_base64_alias():
    payload = ImageGenerationRequest(prompt="test", response_format="base64")

    _validate_image_payload(payload, Settings())

    assert payload.response_format == "b64_json"


def test_response_format_rejects_unknown_values():
    payload = ImageGenerationRequest(prompt="test", response_format="json")

    try:
        _validate_image_payload(payload, Settings())
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "response_format" in exc.detail
    else:
        raise AssertionError("Expected invalid response_format to fail")


def test_seed_defaults_to_42():
    payload = ImageGenerationRequest(prompt="test")

    assert payload.seed == 42


def test_restoration_request_accepts_three_professional_modes():
    for mode in ("preserve", "balanced", "creative", "auto"):
        payload = ImageGenerationRequest(
            prompt="restore",
            image=_sample_base64_png(),
            enhance_mode="restoration",
            restoration_mode=mode,
        )

        _validate_image_payload(payload, Settings())


def test_qwen_image_2512_uses_official_defaults():
    settings = Settings(_env_file=None)

    assert settings.qwen_image_model_path == "Qwen/Qwen-Image-2512"
    assert settings.qwen_image_steps == 50
    assert settings.qwen_image_true_cfg_scale == 4.0
    assert settings.qwen_image_negative_prompt == (
        "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
        "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
    )


def test_qwen_edit_defaults_prioritize_official_high_quality_workflow():
    settings = Settings(_env_file=None)

    assert settings.qwen_edit_steps == 40
    assert settings.qwen_edit_guidance_scale == 1.0
    assert settings.qwen_edit_true_cfg_scale == 4.0
    assert settings.qwen_edit_scale_to_length == 2048
    assert settings.qwen_edit_input_fit_mode == "cover"
    assert settings.qwen_edit_lightning_lora_enabled is False
    assert settings.swinir_enabled is True
    assert settings.codeformer_enabled is True
    assert settings.insightface_enabled is True
    assert settings.qwen_edit_lightning_lora_scale == 1.0
    assert settings.qwen_edit_scheduler_base_shift == pytest.approx(1.0986122886681098)
    assert settings.qwen_unblur_upscale_trigger_prompt == (
        "Unblur and upscale this image. Restore only missing high-frequency details and improve clarity. "
        "Strictly preserve the original identity, facial geometry, facial features, expression, skin tone, "
        "hairstyle, pose, body proportions, clothing, composition, camera angle, lighting, colors, background, "
        "text, and object positions. Do not add, remove, move, reshape, or redesign any subject or object."
    )
    assert settings.qwen_unblur_upscale_alignment_enabled is True
    assert settings.qwen_unblur_upscale_alignment_mode == "similarity"
    assert settings.upscale_fit_mode == "cover"


def test_qwen_edit_applies_lightning_lora_and_scheduler():
    class FakeScheduler:
        config = {"existing": True}

        @classmethod
        def from_config(cls, config, **kwargs):
            instance = cls()
            instance.applied = kwargs
            return instance

    class FakePipeline:
        def __init__(self):
            self.scheduler = FakeScheduler()
            self.loaded = []
            self.adapters = None

        def load_lora_weights(self, path, **kwargs):
            self.loaded.append((path, kwargs))

        def set_adapters(self, names, adapter_weights):
            self.adapters = (names, adapter_weights)

    settings = Settings(_env_file=None, qwen_edit_lightning_lora_enabled=True)
    service = QwenImageEditService(settings)
    pipe = FakePipeline()

    service._apply_edit_scheduler(pipe)
    service._apply_edit_lora(pipe)

    assert pipe.scheduler.applied["base_shift"] == pytest.approx(settings.qwen_edit_scheduler_base_shift)
    assert pipe.scheduler.applied["max_shift"] == pytest.approx(settings.qwen_edit_scheduler_base_shift)
    assert pipe.loaded[0][0] == settings.qwen_edit_lightning_lora_path
    assert pipe.loaded[0][1]["weight_name"] == settings.qwen_edit_lightning_lora_weight_name
    assert pipe.adapters == (["qwen_edit_lightning"], [1.0])


def test_qwen_edit_does_not_replace_scheduler_when_lightning_is_disabled():
    class FakeScheduler:
        config = {"existing": True}

        @classmethod
        def from_config(cls, config, **kwargs):
            raise AssertionError("The scheduler must remain unchanged without Lightning")

    class FakePipeline:
        def __init__(self):
            self.scheduler = FakeScheduler()

    settings = Settings(_env_file=None, qwen_edit_lightning_lora_enabled=False)
    service = QwenImageEditService(settings)
    pipe = FakePipeline()
    original_scheduler = pipe.scheduler

    service._configure_edit_pipeline(pipe)

    assert pipe.scheduler is original_scheduler


def test_qwen_edit_restores_base_scheduler_after_lightning_is_disabled():
    class FakeScheduler:
        def __init__(self, config=None):
            self.config = config or {"base": True}

        @classmethod
        def from_config(cls, config, **kwargs):
            return cls({**dict(config), **kwargs})

    class FakePipeline:
        def __init__(self):
            self.scheduler = FakeScheduler()

    settings = Settings(_env_file=None, qwen_edit_lightning_lora_enabled=True)
    service = QwenImageEditService(settings)
    pipe = FakePipeline()
    service._apply_edit_scheduler(pipe)

    service.settings.qwen_edit_lightning_lora_enabled = False
    service._configure_edit_pipeline(pipe)

    assert pipe.scheduler.config == {"base": True}
    assert service._lightning_scheduler_active is False


def test_qwen_edit_reloads_adapters_when_switching_from_unblur_to_plain_edit():
    class FakePipeline:
        def __init__(self):
            self.loaded = []
            self.unload_count = 0
            self.adapters = []

        def load_lora_weights(self, path, **kwargs):
            self.loaded.append((path, kwargs))

        def set_adapters(self, names, adapter_weights):
            self.adapters.append((names, adapter_weights))

        def unload_lora_weights(self):
            self.unload_count += 1

    service = QwenImageEditService(Settings(_env_file=None, qwen_edit_lightning_lora_enabled=False))
    pipe = FakePipeline()
    service._lora_path = "unblur-lora"
    service._lora_weight_name = "unblur.safetensors"
    service._lora_scale = 0.15
    service._apply_edit_lora(pipe)

    service._lora_path = None
    service._lora_weight_name = None
    service._apply_edit_lora(pipe)

    assert pipe.loaded == [
        (
            "unblur-lora",
            {"adapter_name": "qwen_edit_unblur", "weight_name": "unblur.safetensors"},
        )
    ]
    assert pipe.unload_count == 1
    assert service._edit_lora_active is False
    assert service._active_adapter_key is not None


def test_qwen_edit_injects_pre_sharded_large_components_without_pipeline_remap(monkeypatch):
    class FakePipeline:
        components = {}
        hf_device_map = None

        @classmethod
        def from_pretrained(cls, _model_path, **kwargs):
            cls.load_kwargs = kwargs
            return cls()

        def set_progress_bar_config(self, **_kwargs):
            pass

    class FakeModelManager:
        def unload_except(self, _model_name):
            pass

        def register_model(self, *_args, **_kwargs):
            pass

        def activate_model(self, _model_name):
            pass

    total_budget = {0: "22GiB", 1: "22GiB", 2: "22GiB", 3: "22GiB"}
    class FakeShardedComponent:
        def __init__(self, device_map):
            self.hf_device_map = device_map

    sharded_transformer = FakeShardedComponent(
        {"transformer_blocks.0": 0, "transformer_blocks.1": 1}
    )
    sharded_text_encoder = FakeShardedComponent(
        {"model.layers.0": 2, "model.layers.1": 3}
    )
    settings = Settings(
        _env_file=None,
        device="cuda",
        model_gpu_count=4,
        qwen_edit_pipeline_class="QwenImageEditPlusPipeline",
    )
    service = QwenImageEditService(settings)
    service._model_manager = FakeModelManager()
    monkeypatch.setattr(diffusers, "QwenImageEditPlusPipeline", FakePipeline)
    monkeypatch.setattr(
        "app.qwen_edit_service.get_pipeline_device_map_kwargs",
        lambda *_args: {"device_map": "balanced", "max_memory": total_budget.copy()},
    )
    monkeypatch.setattr(service, "_load_sharded_transformer", lambda *_args: sharded_transformer)
    monkeypatch.setattr(service, "_load_sharded_text_encoder", lambda *_args: sharded_text_encoder)
    monkeypatch.setattr(service, "_move_unsharded_components_to_device", lambda *_args: None)
    monkeypatch.setattr(service, "_configure_edit_pipeline", lambda *_args: None)

    service._get_pipeline()

    assert FakePipeline.load_kwargs["transformer"] is sharded_transformer
    assert FakePipeline.load_kwargs["text_encoder"] is sharded_text_encoder
    assert "device_map" not in FakePipeline.load_kwargs
    assert "max_memory" not in FakePipeline.load_kwargs
    assert service._pipe.hf_device_map == {
        "transformer.transformer_blocks.0": 0,
        "transformer.transformer_blocks.1": 1,
        "text_encoder.model.layers.0": 2,
        "text_encoder.model.layers.1": 3,
    }


def test_qwen_edit_moves_unsharded_vae_to_pipeline_execution_device():
    class FakeModule:
        def __init__(self, *, device="cpu", device_map=None):
            self._device = device
            self.hf_device_map = device_map
            self.moved_to = None

        def parameters(self):
            yield type("Parameter", (), {"device": self._device})()

        def to(self, device):
            self.moved_to = str(device)
            self._device = str(device)
            return self

    class FakeTorch:
        class nn:
            Module = FakeModule

    vae = FakeModule()
    transformer = FakeModule(device="cuda:1", device_map={"block": 1})
    pipe = type(
        "Pipeline",
        (),
        {
            "components": {"transformer": transformer, "vae": vae},
            "_execution_device": "cuda:0",
        },
    )()
    service = QwenImageEditService(Settings(_env_file=None, device="cuda"))
    service._device = "cuda"

    service._move_unsharded_components_to_device(pipe, FakeTorch)

    assert vae.moved_to == "cuda:0"
    assert transformer.moved_to is None


def test_model_manager_distinguishes_device_map_from_cpu_offload():
    class FakePipeline:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(device)

    manager = ModelManager()
    pipe = FakePipeline()

    manager.register_model("mapped", pipe, 1.0, device_mapped=True)
    assert manager.activate_model("mapped") is True

    assert pipe.moves == []
    assert "mapped" in manager._device_mapped_models
    assert "mapped" not in manager._cpu_offload_models


def test_qwen_unblur_lora_settings_respect_enabled_switch():
    disabled = Settings(_env_file=None, qwen_unblur_upscale_lora_enabled=False)
    enabled = Settings(_env_file=None, qwen_unblur_upscale_lora_enabled=True)

    assert _resolve_qwen_unblur_lora(True, disabled) == (None, None)
    assert _resolve_qwen_unblur_lora(False, enabled) == (None, None)
    assert _resolve_qwen_unblur_lora(True, enabled) == (
        enabled.qwen_unblur_upscale_lora_path,
        enabled.qwen_unblur_upscale_lora_weight_name,
    )


def test_qwen_image_uses_fixed_negative_prompt_from_settings():
    settings = Settings(_env_file=None, qwen_image_negative_prompt="  固定中文负面提示词  ")

    assert _resolve_negative_prompt(settings) == "固定中文负面提示词"


def test_qwen_edit_scale_to_length_zero_disables_scaling():
    settings = Settings(_env_file=None, qwen_edit_scale_to_length=0, qwen_edit_round_to_multiple=16)

    assert _resolve_qwen_edit_dimensions(1025, 769, settings) == (1024, 768)


def test_qwen_edit_scale_to_length_rejects_negative_values():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qwen_edit_scale_to_length=-1)


def test_qwen_edit_cover_fit_fills_target_without_black_bars():
    settings = Settings(_env_file=None, qwen_edit_input_fit_mode="cover")
    service = QwenImageEditService(settings)
    image = Image.new("RGB", (200, 100), "red")

    prepared = service._prepare_image(image, 100, 100)

    assert prepared.size == (100, 100)
    assert prepared.getbbox() == (0, 0, 100, 100)
    assert prepared.getpixel((50, 0)) == (255, 0, 0)


def test_qwen_edit_alignment_corrects_enhanced_image_translation():
    settings = Settings(_env_file=None, qwen_unblur_upscale_alignment_enabled=True)
    service = QwenImageEditService(settings)
    reference = Image.new("RGB", (160, 120), "black")
    for x in range(35, 125):
        for y in range(25, 95):
            reference.putpixel((x, y), (220, 180, 120))
    shifted = Image.new("RGB", reference.size, "black")
    shifted.paste(reference.crop((0, 0, 154, 116)), (6, 4))

    aligned = service.align_to_reference(shifted, reference)

    assert aligned.size == reference.size
    assert aligned.getpixel((35, 25)) == reference.getpixel((35, 25))
    assert aligned.getpixel((124, 94)) == reference.getpixel((124, 94))


def test_qwen_unblur_alignment_defaults_to_translation_without_local_warp():
    settings = Settings(_env_file=None)

    assert settings.qwen_unblur_upscale_alignment_mode == "similarity"
    assert settings.qwen_unblur_upscale_alignment_max_side == 1024
    assert settings.qwen_unblur_upscale_alignment_flow_strength == 1.0


def test_qwen_unblur_similarity_transform_rejects_excessive_scale_or_rotation():
    settings = Settings(
        _env_file=None,
        qwen_unblur_upscale_alignment_max_shift=32,
        qwen_unblur_upscale_alignment_max_scale_delta=0.05,
        qwen_unblur_upscale_alignment_max_rotation_degrees=2.0,
    )
    service = QwenImageEditService(settings)

    assert service._similarity_transform_is_safe(scale=1.03, rotation_degrees=1.5, shift_x=8, shift_y=-6)
    assert not service._similarity_transform_is_safe(scale=1.08, rotation_degrees=1.5, shift_x=8, shift_y=-6)
    assert not service._similarity_transform_is_safe(scale=1.03, rotation_degrees=3.0, shift_x=8, shift_y=-6)


def test_upscale_cover_fit_fills_target_without_black_bars():
    settings = Settings(_env_file=None, upscale_fit_mode="cover", upscale_fill_color="black")
    service = ImageUpscaleService(settings)
    image = Image.new("RGB", (200, 100), "red")

    fitted = service._fit_to_target(image, width=100, height=100, fit_mode="cover")

    assert fitted.size == (100, 100)
    assert fitted.getbbox() == (0, 0, 100, 100)
    assert fitted.getpixel((50, 0)) == (255, 0, 0)


def test_upscale_contain_preserves_full_image_with_letterbox():
    settings = Settings(_env_file=None, upscale_fill_color="black")
    service = ImageUpscaleService(settings)
    image = Image.new("RGB", (200, 100), "red")

    fitted = service._fit_to_target(image, width=100, height=100, fit_mode="contain")

    assert fitted.getbbox() == (0, 25, 100, 75)


def test_realesrgan_enhance_mode_allows_empty_model_path_for_default_weights():
    payload = ImageGenerationRequest(prompt="test", image=_sample_base64_png(), enhance_mode="realesrgan")

    _validate_image_payload(payload, Settings(service_role="api", realesrgan_model_path=""))


def test_image_request_accepts_two_image_values():
    encoded = _sample_base64_png()
    payload = ImageGenerationRequest(prompt="merge", model="qwen-image-2512", image=[encoded, encoded])

    _validate_image_payload(payload, Settings())

    reference_image = _payload_image_to_reference(payload.image)
    assert isinstance(reference_image, list)
    assert len(reference_image) == 2
    assert _resolve_enhance_mode(payload, Settings()) == "qwen_edit"


def test_image_request_rejects_three_image_values():
    encoded = _sample_base64_png()
    payload = ImageGenerationRequest(prompt="merge", image=[encoded, encoded, encoded])

    try:
        _validate_image_payload(payload, Settings())
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "one or two" in exc.detail
    else:
        raise AssertionError("Expected more than two image values to fail")


def test_generation_dimensions_use_reference_aspect_ratio_when_missing():
    payload = ImageGenerationRequest(prompt="test", image=_sample_base64_png())
    reference_image = Image.new("RGB", (1600, 900), "red")

    assert _resolve_dimensions(payload, Settings(), reference_image) == (1664, 928)


def test_unblur_upscale_without_resolution_preserves_exact_reference_aspect_ratio():
    payload = ImageGenerationRequest(
        prompt="unblur and upscale",
        image=_sample_base64_png(),
        enhance_mode="qwen_unblur_upscale_realesrgan",
    )
    reference_image = Image.new("RGB", (1920, 1200), "red")

    width, height = _resolve_dimensions(payload, Settings(), reference_image)

    assert width % 16 == 0
    assert height % 16 == 0
    assert abs((width / height) - (1920 / 1200)) < 0.002


def test_edit_dimensions_use_reference_aspect_ratio_when_missing():
    payload = ImageGenerationRequest(endpoint="edits", prompt="test", image=_sample_base64_png())
    reference_image = Image.new("RGB", (900, 1600), "red")

    width, height = _resolve_dimensions(payload, Settings(), reference_image)

    assert height > width
    assert width % 16 == 0
    assert height % 16 == 0
    assert abs((width / height) - (9 / 16)) < 0.02


def test_edit_resolution_uses_reference_aspect_ratio_when_missing():
    payload = ImageGenerationRequest(endpoint="edits", prompt="test", image=_sample_base64_png(), resolution="2k")
    reference_image = Image.new("RGB", (1600, 900), "red")

    assert _resolve_dimensions(payload, Settings(), reference_image) == (2560, 1440)


def test_edit_4k_resolution_preserves_reference_aspect_ratio_when_missing():
    payload = ImageGenerationRequest(endpoint="edits", prompt="test", image=_sample_base64_png(), resolution="4k")
    reference_image = Image.new("RGB", (1000, 1500), "red")

    width, height = _resolve_dimensions(payload, Settings(), reference_image)

    assert (width, height) == (2736, 4096)
    assert abs((width / height) - (1000 / 1500)) < 0.002


def test_unblur_upscale_4k_generation_preserves_reference_aspect_ratio():
    payload = ImageGenerationRequest(
        endpoint="generations",
        prompt="unblur and upscale",
        image=_sample_base64_png(),
        enhance_mode="qwen_unblur_upscale_realesrgan",
        resolution="4k",
    )
    reference_image = Image.new("RGB", (1920, 1200), "red")

    width, height = _resolve_dimensions(payload, Settings(), reference_image)

    assert (width, height) == (4096, 2560)
    assert width / height == reference_image.width / reference_image.height


def test_explicit_edit_dimensions_override_reference_aspect_ratio():
    payload = ImageGenerationRequest(endpoint="edits", prompt="test", image=_sample_base64_png(), size="1024x768")
    reference_image = Image.new("RGB", (900, 1600), "red")

    assert _resolve_dimensions(payload, Settings(), reference_image) == (1024, 768)


def test_face_enhance_request_overrides_settings():
    assert _resolve_face_enhance(ImageGenerationRequest(prompt="test", face_enhance=True), Settings(realesrgan_face_enhance=False)) is True
    assert _resolve_face_enhance(ImageGenerationRequest(prompt="test", face_enhance=False), Settings(realesrgan_face_enhance=True)) is False


def test_face_enhance_can_be_read_from_prompt_params():
    payload = ImageGenerationRequest(prompt="restore portrait [face_enhance=true]")

    _apply_prompt_params(payload)

    assert payload.face_enhance is True
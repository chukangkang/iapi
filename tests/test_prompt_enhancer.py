import json

import pytest

from app.config import Settings
from app.main import ImageGenerationRequest, _enhance_image_prompt
from app.prompt_enhancer import PromptEnhancer


class FakeHttpResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self._body


def test_chat_interface_builds_request_and_extracts_expanded_prompt(monkeypatch, caplog):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeHttpResponse(
            {"choices": [{"message": {"content": '{"expanded_prompt":"电影感海边日落场景"}'}}]}
        )

    monkeypatch.setattr("app.prompt_enhancer.urlopen", fake_urlopen)
    settings = Settings(
        _env_file=None,
        prompt_enhancer_base_url="http://text.example/v1/",
        prompt_enhancer_api_key="secret",
        prompt_enhancer_model="qwen3.6-27b",
        prompt_enhancer_api_type="chat",
    )

    with caplog.at_level("INFO"):
        result = PromptEnhancer(settings)._enhance_sync("女孩在海边", aspect_ratio="16:9")

    assert result == "电影感海边日落场景"
    assert captured["url"] == "http://text.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["payload"]["model"] == "qwen3.6-27b"
    assert captured["payload"]["messages"][-1]["content"] == "用户提示词：女孩在海边\n画面比例：16:9"
    assert "Calling prompt enhancer: api_type=chat model=qwen3.6-27b" in caplog.text
    assert "Prompt enhancer completed: model=qwen3.6-27b" in caplog.text


def test_responses_interface_extracts_nested_output_text(monkeypatch):
    def fake_urlopen(request, timeout):
        payload = json.loads(request.data)
        assert request.full_url == "http://text.example/v1/responses"
        assert payload["input"] == "用户提示词：一只猫"
        return FakeHttpResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "```json\n{\"expanded_prompt\":\"窗边的橘猫\"}\n```"}
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("app.prompt_enhancer.urlopen", fake_urlopen)
    settings = Settings(
        _env_file=None,
        prompt_enhancer_base_url="http://text.example/v1",
        prompt_enhancer_model="qwen3.6-27b",
        prompt_enhancer_api_type="responses",
    )

    assert PromptEnhancer(settings)._enhance_sync("一只猫") == "窗边的橘猫"


@pytest.mark.asyncio
async def test_enhancer_failure_falls_back_to_original_prompt(monkeypatch):
    def fail_urlopen(request, timeout):
        raise OSError("service unavailable")

    monkeypatch.setattr("app.prompt_enhancer.urlopen", fail_urlopen)
    settings = Settings(
        _env_file=None,
        prompt_enhancer_enabled=True,
        prompt_enhancer_base_url="http://text.example/v1",
        prompt_enhancer_model="qwen3.6-27b",
        prompt_enhancer_fallback_to_original=True,
    )

    assert await PromptEnhancer(settings).enhance("原始提示词") == "原始提示词"


def test_enhancer_rejects_missing_required_configuration():
    settings = Settings(_env_file=None, prompt_enhancer_enabled=True)

    with pytest.raises(ValueError, match="BASE_URL"):
        PromptEnhancer(settings).validate_configuration()


@pytest.mark.asyncio
async def test_image_prompt_is_expanded_before_queue_submission(monkeypatch, caplog):
    class FakeEnhancer:
        async def enhance(self, prompt, *, aspect_ratio=None):
            assert prompt == "女孩在海边"
            assert aspect_ratio == "16:9"
            return "傍晚海边，女孩站在潮湿沙滩上，柔和逆光，横向电影构图。"

    monkeypatch.setattr("app.main._get_prompt_enhancer", lambda settings: FakeEnhancer())
    payload = ImageGenerationRequest(prompt="女孩在海边", aspect_ratio="16:9")
    settings = Settings(_env_file=None, prompt_enhancer_enabled=True)

    with caplog.at_level("INFO"):
        await _enhance_image_prompt(payload, settings)

    assert payload.original_prompt == "女孩在海边"
    assert payload.prompt == "傍晚海边，女孩站在潮湿沙滩上，柔和逆光，横向电影构图。"
    assert "Final image prompt after enhancement: 傍晚海边，女孩站在潮湿沙滩上，柔和逆光，横向电影构图。" in caplog.text


@pytest.mark.asyncio
async def test_request_can_disable_prompt_enhancement(monkeypatch):
    def unexpected_enhancer(settings):
        raise AssertionError("enhancer should not be called")

    monkeypatch.setattr("app.main._get_prompt_enhancer", unexpected_enhancer)
    payload = ImageGenerationRequest(prompt="保留原文", prompt_enhance=False)
    settings = Settings(_env_file=None, prompt_enhancer_enabled=True)

    await _enhance_image_prompt(payload, settings)

    assert payload.original_prompt is None
    assert payload.prompt == "保留原文"

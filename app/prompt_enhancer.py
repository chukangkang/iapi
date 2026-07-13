import asyncio
import json
import logging
import re
import time
from typing import Any, Optional
from urllib.request import Request, urlopen

from app.config import Settings


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 Qwen-Image 图像生成提示词优化器。
将用户输入扩写为准确、自然、层次清晰的中文视觉描述。
必须保留用户指定的主体、身份、数量、动作、文字原文、风格、颜色与构图，不得改变原意。
仅补充可观察、可绘制的场景、空间关系、材质、光照、镜头、景深和色彩信息。
不得添加原提示词中不存在的人物、物体、品牌、文字或剧情。
不要输出负面提示词，不要堆砌“杰作、8K、最佳质量”等标签。
扩写结果控制在 300 至 600 个中文字符以内；简单场景可更短。
只输出扩写后的完整提示词正文，不要输出 JSON、标题、解释、思考过程或 Markdown。"""


class PromptEnhancer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate_configuration(self) -> None:
        if not self.settings.prompt_enhancer_enabled:
            return
        if not self.settings.prompt_enhancer_base_url.strip():
            raise ValueError("PROMPT_ENHANCER_BASE_URL is required when prompt enhancement is enabled")
        if not self.settings.prompt_enhancer_model.strip():
            raise ValueError("PROMPT_ENHANCER_MODEL is required when prompt enhancement is enabled")

    async def enhance(self, prompt: str, *, aspect_ratio: Optional[str] = None) -> str:
        try:
            return await asyncio.to_thread(self._enhance_sync, prompt, aspect_ratio=aspect_ratio)
        except Exception:
            if not self.settings.prompt_enhancer_fallback_to_original:
                raise
            logger.exception("Prompt enhancement failed; falling back to the original prompt")
            return prompt

    def _enhance_sync(self, prompt: str, *, aspect_ratio: Optional[str] = None) -> str:
        self._validate_request_configuration()
        user_input = self._build_user_input(prompt, aspect_ratio)
        api_type = self.settings.prompt_enhancer_api_type
        payload = self._build_chat_payload(user_input) if api_type == "chat" else self._build_responses_payload(user_input)
        endpoint = "chat/completions" if api_type == "chat" else "responses"
        url = f"{self.settings.prompt_enhancer_base_url.rstrip('/')}/{endpoint}"
        logger.info(
            "Calling prompt enhancer: api_type=%s model=%s endpoint=%s",
            api_type,
            self.settings.prompt_enhancer_model,
            url,
        )
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        started = time.monotonic()
        with urlopen(request, timeout=self.settings.prompt_enhancer_timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = self._extract_chat_text(body) if api_type == "chat" else self._extract_responses_text(body)
        if body.get("status") == "incomplete":
            details = body.get("incomplete_details") or {}
            logger.warning(
                "Prompt enhancer returned partial output: status=incomplete reason=%s",
                details.get("reason", "unknown"),
            )
        expanded_prompt = self._parse_expanded_prompt(content)
        if not expanded_prompt:
            raise ValueError("Prompt enhancer returned an empty expanded_prompt")
        logger.info(
            "Prompt enhancer completed: model=%s elapsed_ms=%s prompt_chars=%s",
            self.settings.prompt_enhancer_model,
            round((time.monotonic() - started) * 1000),
            len(expanded_prompt),
        )
        return expanded_prompt

    def _validate_request_configuration(self) -> None:
        if not self.settings.prompt_enhancer_base_url.strip():
            raise ValueError("PROMPT_ENHANCER_BASE_URL is required")
        if not self.settings.prompt_enhancer_model.strip():
            raise ValueError("PROMPT_ENHANCER_MODEL is required")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.settings.prompt_enhancer_api_key:
            headers["Authorization"] = f"Bearer {self.settings.prompt_enhancer_api_key}"
        return headers

    def _build_chat_payload(self, user_input: str) -> dict[str, Any]:
        return {
            "model": self.settings.prompt_enhancer_model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_input},
            ],
            "temperature": self.settings.prompt_enhancer_temperature,
            "max_tokens": self.settings.prompt_enhancer_max_tokens,
            "enable_thinking": False,
            "stream": False,
        }

    def _build_responses_payload(self, user_input: str) -> dict[str, Any]:
        return {
            "model": self.settings.prompt_enhancer_model,
            "instructions": self._system_prompt(),
            "input": user_input,
            "temperature": self.settings.prompt_enhancer_temperature,
            "max_output_tokens": self.settings.prompt_enhancer_max_tokens,
            "reasoning": {"effort": "none"},
            "enable_thinking": False,
            "text": {"verbosity": "low"},
            "stream": False,
        }

    def _system_prompt(self) -> str:
        region = " ".join(self.settings.prompt_enhancer_default_person_region.split())
        if not region:
            return SYSTEM_PROMPT
        return (
            f"{SYSTEM_PROMPT}\n"
            f"当画面包含人物，且用户没有明确指定国家、地区、族裔或人物地域外貌时，默认描述为{region}人物，"
            f"使用自然、真实、多样的{region}人物面部特征；不得刻板化或夸张化。"
            "不得覆盖用户明确指定的国家、地区、族裔或人物身份；没有人物时不得添加人物。"
        )

    @staticmethod
    def _build_user_input(prompt: str, aspect_ratio: Optional[str]) -> str:
        value = f"用户提示词：{prompt.strip()}"
        if aspect_ratio:
            value += f"\n画面比例：{aspect_ratio.strip()}"
        return value

    @staticmethod
    def _extract_chat_text(body: dict[str, Any]) -> str:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Invalid Chat Completions response") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        raise ValueError("Chat Completions response contains no text")

    @staticmethod
    def _extract_responses_text(body: dict[str, Any]) -> str:
        if isinstance(body.get("output_text"), str):
            return body["output_text"]
        texts = []
        for output in body.get("output", []):
            if not isinstance(output, dict):
                continue
            for content in output.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    texts.append(str(content.get("text", "")))
        if not texts:
            raise ValueError("Invalid Responses API response")
        return "".join(texts)

    @staticmethod
    def _parse_expanded_prompt(content: str) -> str:
        value = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            value = fenced.group(1)
        if not value:
            return ""
        if not value.startswith("{"):
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            truncated = re.match(r'^\s*\{\s*"expanded_prompt"\s*:\s*"(.*)$', value, flags=re.DOTALL)
            if truncated:
                return truncated.group(1).rstrip('"}').strip()
            raise ValueError("Prompt enhancer returned malformed JSON without a recoverable expanded_prompt")
        if not isinstance(parsed, dict) or not isinstance(parsed.get("expanded_prompt"), str):
            raise ValueError("Prompt enhancer JSON is missing expanded_prompt")
        return parsed["expanded_prompt"].strip()
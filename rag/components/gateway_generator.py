"""Chat generator 元件:公司閘道(OpenAI 相容)與離線 mock。

兩者都符合 Haystack 的 ChatGenerator 形狀(``run(messages) →
{"replies": [ChatMessage]}``),因此除了 generation 槽位,也能直接
插進 ``LLMRanker`` 與 ``LLMQueryDecomposer``。
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

from haystack import component
from haystack.dataclasses import ChatMessage

from rag.errors import APICallError, APIResponseFormatError

logger = logging.getLogger(__name__)

# OpenAI 推理模型系列(gpt-5 / gpt-5.1 / o1 / o3 / o4 …)的取樣參數是固定的:
# temperature 只接受預設值 1,帶其他值會直接回 400;token 上限也改用
# max_completion_tokens 欄位。gpt-5-chat-latest 之類的一般 chat 模型不在此列。
_REASONING_MODEL_PATTERN = re.compile(r"(gpt-5|o[1-9])([.-]|$)")


def is_reasoning_model(model: str | None) -> bool:
    """判斷是否為只接受預設取樣參數的 OpenAI 推理模型。"""
    if not model:
        return False
    name = model.lower()
    if "chat" in name:
        return False
    return _REASONING_MODEL_PATTERN.match(name) is not None


@component
class GatewayChatGenerator:
    """OpenAI 相容 chat completions 客戶端,為內部閘道保留兩個關鍵行為:

    - ``model`` 未設定時,請求體**完全不帶** model 欄位
      (公司閘道收到未知欄位會拒絕;官方 OpenAI SDK 做不到這一點)。
    - 推理模型自動忽略 temperature、以 ``max_completion_tokens`` 送出
      token 上限,不需要調整 YAML。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
        completions_path: str = "/chat/completions",
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._reasoning_model = is_reasoning_model(model)
        if self._reasoning_model and temperature is not None:
            logger.warning(
                "模型 %s 只接受預設 temperature,已忽略設定值 %s", model, temperature
            )
        self.endpoint = base_url.rstrip("/") + completions_path
        self.headers = dict(headers or {})
        if api_key:
            self.headers.setdefault("Authorization", f"Bearer {api_key}")
        self._client = client  # 可注入以便測試

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """呼叫 chat completions API 並回傳單一回覆。

        Raises:
            APICallError: timeout、連線失敗或非 2xx 狀態碼。
            APIResponseFormatError: 回應結構不符 OpenAI chat 格式。
        """
        body: dict[str, Any] = {
            "messages": [
                {"role": message.role.value, "content": message.text or ""}
                for message in messages
            ]
        }
        if self.model is not None:
            body["model"] = self.model
        if self.temperature is not None and not self._reasoning_model:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            key = "max_completion_tokens" if self._reasoning_model else "max_tokens"
            body[key] = self.max_tokens

        started = time.perf_counter()
        data = self._request_json(body)
        latency_ms = (time.perf_counter() - started) * 1000
        answer, finish_reason = self._extract_answer(data)

        meta: dict[str, Any] = {"model": self.model, "latency_ms": latency_ms}
        if finish_reason is not None:
            meta["finish_reason"] = finish_reason
        if isinstance(data, dict) and isinstance(data.get("usage"), dict):
            meta["usage"] = data["usage"]  # 原始 usage 完整保留(供應商專屬欄位)
        return {"replies": [ChatMessage.from_assistant(answer, meta=meta)]}

    def _request_json(self, body: dict[str, Any]) -> Any:
        try:
            if self._client is not None:
                response = self._client.post(
                    self.endpoint, json=body, headers=self.headers, timeout=self.timeout
                )
            else:
                response = httpx.post(
                    self.endpoint, json=body, headers=self.headers, timeout=self.timeout
                )
        except httpx.TimeoutException as exc:
            raise APICallError(
                f"呼叫 API 逾時({self.timeout} 秒):POST {self.endpoint}"
            ) from exc
        except httpx.HTTPError as exc:
            raise APICallError(f"呼叫 API 失敗:POST {self.endpoint}({exc})") from exc
        if not 200 <= response.status_code < 300:
            preview = response.text[:200]
            raise APICallError(
                f"API 回應非 2xx 狀態碼 {response.status_code}:POST {self.endpoint}"
                f"(回應內容前 200 字:{preview!r})"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise APIResponseFormatError(
                f"API 回應不是合法的 JSON:POST {self.endpoint}"
            ) from exc

    @staticmethod
    def _extract_answer(data: Any) -> tuple[str, str | None]:
        """從回應中取出 ``choices[0].message.content`` 與 finish_reason。

        錯誤訊息會列出實際存在的欄位;相容舊式 completions 回應
        (``choices[0].text``)。
        """
        if not isinstance(data, dict) or "choices" not in data:
            hint = (
                f"實際的欄位:{sorted(data.keys())}"
                if isinstance(data, dict)
                else f"實際型別:{type(data).__name__}"
            )
            raise APIResponseFormatError(f"LLM API 回應缺少 'choices' 欄位;{hint}")
        choices = data["choices"]
        if not isinstance(choices, list) or not choices:
            raise APIResponseFormatError("LLM API 回應的 'choices' 必須是非空 list")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise APIResponseFormatError(
                f"choices[0] 必須是物件,實際得到:{type(choice).__name__}"
            )
        message = choice.get("message")
        content: Any = None
        if isinstance(message, dict):
            content = message.get("content")
        if content is None:
            content = choice.get("text")
        if not isinstance(content, str):
            raise APIResponseFormatError(
                "LLM API 回應中找不到回答文字(choices[0].message.content 或 "
                f"choices[0].text);choices[0] 實際的欄位:{sorted(choice.keys())}"
            )
        finish_reason = choice.get("finish_reason")
        return content, finish_reason if isinstance(finish_reason, str) else None


@component
class MockChatGenerator:
    """離線 mock chat generator:不接任何 LLM。

    ``replies`` 有設定時依序循環回覆(供測試腳本化,例如 LLMRanker 的
    JSON 排序回覆);未設定時回覆一段可辨識的假答案,內容引用最後一則
    user 訊息的開頭,方便肉眼確認 prompt 有正確組出來。
    """

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self._cursor = 0

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        if self.replies:
            text = self.replies[self._cursor % len(self.replies)]
            self._cursor += 1
        else:
            last_user = next(
                (m.text or "" for m in reversed(messages) if m.role.value == "user"),
                "",
            )
            snippet = " ".join(last_user.split())[:60]
            text = f"[mock 回答] 根據提供的內容回應(prompt 開頭:{snippet}…)"
        meta = {"model": "mock", "finish_reason": "stop"}
        return {"replies": [ChatMessage.from_assistant(text, meta=meta)]}

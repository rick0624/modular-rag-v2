"""GatewayChatGenerator 測試:閘道 quirks 與回應診斷。"""

from __future__ import annotations

import pytest
from conftest import FakeClient, FakeResponse

from haystack.dataclasses import ChatMessage

from rag.components.gateway_generator import (
    GatewayChatGenerator,
    MockChatGenerator,
    is_reasoning_model,
)
from rag.errors import APIResponseFormatError

BASE_URL = "https://llm.example.com/v1"
OK_PAYLOAD = {
    "choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _generator(client, **kwargs):
    return GatewayChatGenerator(base_url=BASE_URL, client=client, **kwargs)


def _run(generator):
    return generator.run(messages=[ChatMessage.from_user("問題")])


class TestReasoningModelDetection:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-5", True),
            ("gpt-5-nano", True),
            ("gpt-5.1", True),
            ("o3-mini", True),
            ("o1", True),
            ("gpt-5-chat-latest", False),  # 名稱含 chat 的一般模型
            ("gpt-4o", False),
            ("llama-3", False),
            (None, False),
        ],
    )
    def test_detection(self, model, expected):
        assert is_reasoning_model(model) is expected


class TestRequestBody:
    def test_model_omitted_when_none(self):
        client = FakeClient([FakeResponse(payload=OK_PAYLOAD)])
        _run(_generator(client, temperature=0.2))
        body = client.calls[0]["json"]
        assert "model" not in body, "內部閘道模式:不帶 model 欄位"
        assert body["temperature"] == 0.2
        assert body["messages"] == [{"role": "user", "content": "問題"}]

    def test_reasoning_model_drops_temperature_and_renames_max_tokens(self):
        client = FakeClient([FakeResponse(payload=OK_PAYLOAD)])
        _run(_generator(client, model="gpt-5-nano", temperature=0.7, max_tokens=100))
        body = client.calls[0]["json"]
        assert body["model"] == "gpt-5-nano"
        assert "temperature" not in body
        assert "max_tokens" not in body
        assert body["max_completion_tokens"] == 100

    def test_normal_model_keeps_max_tokens(self):
        client = FakeClient([FakeResponse(payload=OK_PAYLOAD)])
        _run(_generator(client, model="gpt-5-chat-latest", max_tokens=64))
        body = client.calls[0]["json"]
        assert body["max_tokens"] == 64
        assert "max_completion_tokens" not in body

    def test_api_key_becomes_bearer_header(self):
        client = FakeClient([FakeResponse(payload=OK_PAYLOAD)])
        _run(_generator(client, api_key="secret"))
        assert client.calls[0]["headers"]["Authorization"] == "Bearer secret"
        assert client.calls[0]["url"] == f"{BASE_URL}/chat/completions"


class TestResponseHandling:
    def test_reply_text_and_meta(self):
        client = FakeClient([FakeResponse(payload=OK_PAYLOAD)])
        reply = _run(_generator(client))["replies"][0]
        assert reply.text == "答案"
        assert reply.meta["finish_reason"] == "stop"
        assert reply.meta["usage"]["total_tokens"] == 15
        assert reply.meta["latency_ms"] >= 0

    def test_legacy_completions_text_fallback(self):
        client = FakeClient(
            [FakeResponse(payload={"choices": [{"text": "舊式回答"}]})]
        )
        assert _run(_generator(client))["replies"][0].text == "舊式回答"

    def test_missing_choices_lists_actual_fields(self):
        client = FakeClient([FakeResponse(payload={"id": "x", "object": "chat"})])
        with pytest.raises(
            APIResponseFormatError, match=r"實際的欄位:\['id', 'object'\]"
        ):
            _run(_generator(client))

    def test_missing_content_lists_choice_fields(self):
        client = FakeClient(
            [FakeResponse(payload={"choices": [{"finish_reason": "stop"}]})]
        )
        with pytest.raises(
            APIResponseFormatError, match=r"choices\[0\] 實際的欄位:\['finish_reason'\]"
        ):
            _run(_generator(client))


class TestMockChatGenerator:
    def test_scripted_replies_cycle(self):
        generator = MockChatGenerator(replies=["一", "二"])
        texts = [_run(generator)["replies"][0].text for _ in range(3)]
        assert texts == ["一", "二", "一"]

    def test_default_reply_echoes_prompt_head(self):
        generator = MockChatGenerator()
        reply = generator.run(
            messages=[ChatMessage.from_user("根據以下內容回答問題:...")]
        )["replies"][0]
        assert reply.text.startswith("[mock 回答]")
        assert "根據以下內容回答問題" in reply.text

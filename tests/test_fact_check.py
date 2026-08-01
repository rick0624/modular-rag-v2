"""LLMFactChecker 測試:過濾語意、fail-soft、max_docs、prompt 渲染。"""

from __future__ import annotations

import pytest
from haystack import Document

from rag.components.fact_check import LLMFactChecker
from rag.components.gateway_generator import MockChatGenerator


def _docs():
    return [
        Document(content="FAISS 支援多種索引結構", score=0.9),
        Document(content="今天天氣很好", score=0.8),
        Document(content="FAISS 支援 GPU 加速", score=0.7),
    ]


class TestFiltering:
    def test_keeps_listed_indices_in_original_order(self):
        docs = _docs()
        checker = LLMFactChecker(
            chat_generator=MockChatGenerator(replies=['{"relevant_indices": [3, 1]}'])
        )
        out = checker.run(query="FAISS 功能", documents=docs)
        # 保留原輸入順序與原分數(同一批 Document 物件)
        assert out["documents"] == [docs[0], docs[2]]
        assert [d.score for d in out["documents"]] == [0.9, 0.7]

    def test_empty_indices_returns_empty(self):
        checker = LLMFactChecker(
            chat_generator=MockChatGenerator(replies=['{"relevant_indices": []}'])
        )
        assert checker.run(query="q", documents=_docs())["documents"] == []

    def test_invalid_indices_ignored_with_warning(self, caplog):
        checker = LLMFactChecker(
            chat_generator=MockChatGenerator(
                replies=['{"relevant_indices": [0, 99, "x", 2]}']
            )
        )
        docs = _docs()
        with caplog.at_level("WARNING"):
            out = checker.run(query="q", documents=docs)
        assert out["documents"] == [docs[1]]
        assert "無效編號" in caplog.text


class TestFailSoft:
    def test_non_json_reply_keeps_all(self, caplog):
        checker = LLMFactChecker(
            chat_generator=MockChatGenerator(replies=["這不是 JSON"])
        )
        docs = _docs()
        with caplog.at_level("WARNING"):
            out = checker.run(query="q", documents=docs)
        assert out["documents"] == docs
        assert "事實查核" in caplog.text

    def test_generator_exception_keeps_all(self):
        class BrokenGenerator:
            def run(self, messages):
                raise RuntimeError("LLM 掛了")

        checker = LLMFactChecker(chat_generator=BrokenGenerator())
        docs = _docs()
        assert checker.run(query="q", documents=docs)["documents"] == docs

    def test_missing_indices_field_keeps_all(self):
        checker = LLMFactChecker(
            chat_generator=MockChatGenerator(replies=['{"other": 1}'])
        )
        docs = _docs()
        assert checker.run(query="q", documents=docs)["documents"] == docs


class TestEdges:
    def test_empty_input_skips_llm_call(self):
        calls: list[str] = []

        class RecordingGenerator:
            def run(self, messages):
                calls.append(messages[0].text)
                return {"replies": []}

        checker = LLMFactChecker(chat_generator=RecordingGenerator())
        assert checker.run(query="q", documents=[])["documents"] == []
        assert calls == []

    def test_max_docs_tail_passes_through(self):
        docs = _docs()
        checker = LLMFactChecker(
            chat_generator=MockChatGenerator(replies=['{"relevant_indices": [1]}']),
            max_docs=2,
        )
        out = checker.run(query="q", documents=docs)
        # 前 2 筆受檢(只留第 1 筆),第 3 筆未受檢原樣通過
        assert out["documents"] == [docs[0], docs[2]]

    def test_prompt_renders_numbered_chunks_and_query(self):
        received: list[str] = []

        class RecordingGenerator:
            def run(self, messages):
                received.append(messages[0].text)
                return {"replies": []}

        checker = LLMFactChecker(chat_generator=RecordingGenerator())
        checker.run(query="FAISS 功能", documents=_docs())
        prompt = received[0]
        assert "FAISS 功能" in prompt
        assert "1. FAISS 支援多種索引結構" in prompt
        assert "3. FAISS 支援 GPU 加速" in prompt

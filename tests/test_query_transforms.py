"""Query transformation 元件測試:normalize / glossary / llm_decompose。"""

from __future__ import annotations

import pytest

from rag.components.gateway_generator import MockChatGenerator
from rag.components.query_transforms import (
    GlossaryExpander,
    LLMQueryDecomposer,
    QueryNormalizer,
)
from rag.errors import ComponentError


class TestNormalizer:
    def test_nfkc_whitespace_lowercase(self):
        out = QueryNormalizer().run(queries=["  ＦＡＩＳＳ　是   什麼？  "])
        assert out["queries"] == ["faiss 是 什麼?"]

    def test_lowercase_off(self):
        out = QueryNormalizer(lowercase=False).run(queries=["FAISS 是什麼"])
        assert out["queries"] == ["FAISS 是什麼"]

    def test_maps_each_query(self):
        out = QueryNormalizer().run(queries=["A  B", "Ｃ"])
        assert out["queries"] == ["a b", "c"]


class TestGlossary:
    GLOSSARY = {"RRF": "Reciprocal Rank Fusion,名次融合法", "KB": "知識庫"}

    def test_match_produces_notes(self):
        expander = GlossaryExpander(glossary=self.GLOSSARY)
        out = expander.run(queries=["rrf 是怎麼運作的?"])
        assert out["queries"] == ["rrf 是怎麼運作的?"]  # 預設不改查詢
        assert out["notes"] == "RRF:Reciprocal Rank Fusion,名次融合法"

    def test_no_match_empty_notes(self):
        out = GlossaryExpander(glossary=self.GLOSSARY).run(queries=["向量檢索"])
        assert out["notes"] == ""

    def test_expand_query_appends_definitions(self):
        expander = GlossaryExpander(glossary=self.GLOSSARY, expand_query=True)
        out = expander.run(queries=["KB 怎麼建?"])
        assert out["queries"] == ["KB 怎麼建? KB(知識庫)"]

    def test_glossary_file_loading(self, tmp_path):
        path = tmp_path / "glossary.yaml"
        path.write_text("ES: Elasticsearch\n", encoding="utf-8")
        expander = GlossaryExpander(glossary_path=str(path))
        assert expander.run(queries=["es 支援什麼"])["notes"] == "ES:Elasticsearch"

    def test_missing_glossary_rejected(self):
        with pytest.raises(ComponentError, match="glossary_path"):
            GlossaryExpander()

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(ComponentError, match="找不到術語表檔案"):
            GlossaryExpander(glossary_path=str(tmp_path / "nope.yaml"))


class TestDecomposer:
    def test_scripted_decomposition_strips_numbering(self):
        generator = MockChatGenerator(
            replies=["1. FAISS 是什麼?\n2) 它支援哪些索引?\n- 空行之外都算\n\n"]
        )
        decomposer = LLMQueryDecomposer(chat_generator=generator, max_subqueries=4)
        out = decomposer.run(queries=["FAISS 是什麼?支援哪些索引?"])
        assert out["queries"] == [
            "FAISS 是什麼?",
            "它支援哪些索引?",
            "空行之外都算",
        ]

    def test_max_subqueries_cap(self):
        generator = MockChatGenerator(replies=["1. a\n2. b\n3. c\n4. d"])
        decomposer = LLMQueryDecomposer(chat_generator=generator, max_subqueries=2)
        assert decomposer.run(queries=["q"])["queries"] == ["a", "b"]

    def test_blank_output_fails_soft_to_original(self):
        generator = MockChatGenerator(replies=["   \n  "])
        decomposer = LLMQueryDecomposer(chat_generator=generator)
        assert decomposer.run(queries=["原查詢"])["queries"] == ["原查詢"]

    def test_generator_exception_fails_soft_to_original(self):
        class BrokenGenerator:
            def run(self, messages):
                raise RuntimeError("LLM 掛了")

        decomposer = LLMQueryDecomposer(chat_generator=BrokenGenerator())
        assert decomposer.run(queries=["原查詢"])["queries"] == ["原查詢"]

    def test_each_input_query_decomposed_and_flattened(self):
        generator = MockChatGenerator(replies=["1. a1\n2. a2", "1. b1"])
        decomposer = LLMQueryDecomposer(chat_generator=generator)
        assert decomposer.run(queries=["qa", "qb"])["queries"] == ["a1", "a2", "b1"]

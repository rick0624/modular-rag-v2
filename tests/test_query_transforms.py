"""Query transformation 元件測試:normalize / glossary / jargon_mapping /
llm_rewrite / llm_decompose / llm_multi_hyde / preqrag。"""

from __future__ import annotations

import json

import pytest

from rag.components.gateway_generator import MockChatGenerator
from rag.components.query_transforms import (
    GlossaryExpander,
    JargonMapper,
    LLMMultiHyDEExpander,
    LLMQueryDecomposer,
    LLMQueryRewriter,
    PreQRAGDispatcher,
    QueryNormalizer,
)
from rag.errors import ComponentError


class BrokenGenerator:
    def run(self, messages):
        raise RuntimeError("LLM 掛了")


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


class TestJargonMapper:
    def test_cjk_replacement(self):
        mapper = JargonMapper(mapping={"三高": "高血壓、高血糖、高血脂"})
        out = mapper.run(queries=["三高 患者的飲食建議?"])
        assert out["queries"] == ["高血壓、高血糖、高血脂 患者的飲食建議?"]

    def test_longest_key_wins(self):
        mapper = JargonMapper(
            mapping={"向量": "數值陣列", "向量檢索": "以語意相似度搜尋"}
        )
        out = mapper.run(queries=["向量檢索 與 向量 的差別"])
        assert out["queries"] == ["以語意相似度搜尋 與 數值陣列 的差別"]

    def test_latin_case_insensitive(self):
        mapper = JargonMapper(mapping={"RRF": "多路檢索名次融合"})
        out = mapper.run(queries=["rrf 是什麼?"])
        assert out["queries"] == ["多路檢索名次融合 是什麼?"]

    def test_no_cascading_replacement(self):
        # value 含另一個 key(「檢索」),不可被二次替換
        mapper = JargonMapper(
            mapping={"kNN": "最近鄰檢索", "檢索": "搜尋"}
        )
        out = mapper.run(queries=["kNN 怎麼用?"])
        assert out["queries"] == ["最近鄰檢索 怎麼用?"]

    def test_no_match_passthrough(self):
        mapper = JargonMapper(mapping={"RRF": "多路檢索名次融合"})
        assert mapper.run(queries=["混合檢索是什麼"])["queries"] == ["混合檢索是什麼"]

    def test_json_path_loading(self, tmp_path):
        path = tmp_path / "jargon.json"
        path.write_text(
            json.dumps({"ES": "Elasticsearch 搜尋引擎"}, ensure_ascii=False),
            encoding="utf-8",
        )
        mapper = JargonMapper(json_path=str(path))
        assert mapper.run(queries=["ES 怎麼設定"])["queries"] == [
            "Elasticsearch 搜尋引擎 怎麼設定"
        ]

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(ComponentError, match="找不到術語對照表檔案"):
            JargonMapper(json_path=str(tmp_path / "nope.json"))

    def test_non_dict_json_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('["a", "b"]', encoding="utf-8")
        with pytest.raises(ComponentError, match="扁平 JSON 物件"):
            JargonMapper(json_path=str(path))

    def test_invalid_json_rejected(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ComponentError, match="不是合法的 JSON"):
            JargonMapper(json_path=str(path))

    def test_missing_mapping_rejected(self):
        with pytest.raises(ComponentError, match="json_path"):
            JargonMapper()


class TestLLMRewriter:
    def test_scripted_rewrite(self):
        generator = MockChatGenerator(
            replies=['{"rewritten_query": "FAISS 支援的索引結構", "reason": "去除贅字"}']
        )
        rewriter = LLMQueryRewriter(chat_generator=generator)
        out = rewriter.run(queries=["那個 FAISS 到底支援哪些索引結構啊?"])
        assert out["queries"] == ["FAISS 支援的索引結構"]

    def test_fenced_json_reply(self):
        generator = MockChatGenerator(
            replies=['```json\n{"rewritten_query": "改寫後", "reason": "r"}\n```']
        )
        rewriter = LLMQueryRewriter(chat_generator=generator)
        assert rewriter.run(queries=["原查詢"])["queries"] == ["改寫後"]

    def test_garbage_reply_fails_soft(self, caplog):
        generator = MockChatGenerator(replies=["這不是 JSON"])
        rewriter = LLMQueryRewriter(chat_generator=generator)
        with caplog.at_level("WARNING"):
            out = rewriter.run(queries=["原查詢"])
        assert out["queries"] == ["原查詢"]
        assert "查詢改寫" in caplog.text

    def test_generator_exception_fails_soft(self):
        class BrokenGenerator:
            def run(self, messages):
                raise RuntimeError("LLM 掛了")

        rewriter = LLMQueryRewriter(chat_generator=BrokenGenerator())
        assert rewriter.run(queries=["原查詢"])["queries"] == ["原查詢"]

    def test_blank_rewritten_query_fails_soft(self):
        generator = MockChatGenerator(
            replies=['{"rewritten_query": "  ", "reason": "r"}']
        )
        rewriter = LLMQueryRewriter(chat_generator=generator)
        assert rewriter.run(queries=["原查詢"])["queries"] == ["原查詢"]

    def test_one_to_one_over_multiple_queries(self):
        generator = MockChatGenerator(
            replies=[
                '{"rewritten_query": "改寫 A", "reason": "r"}',
                '{"rewritten_query": "改寫 B", "reason": "r"}',
            ]
        )
        rewriter = LLMQueryRewriter(chat_generator=generator)
        assert rewriter.run(queries=["qa", "qb"])["queries"] == ["改寫 A", "改寫 B"]

    def test_custom_prompt_reaches_generator(self):
        received: list[str] = []

        class RecordingGenerator:
            def run(self, messages):
                received.append(messages[0].text)
                raise RuntimeError("只記錄,不回答")

        rewriter = LLMQueryRewriter(
            chat_generator=RecordingGenerator(), prompt="自訂:{{ query }}"
        )
        rewriter.run(queries=["問題文字"])
        assert received == ["自訂:問題文字"]


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


class TestMultiHyDE:
    def test_original_plus_hypothetical_documents(self):
        generator = MockChatGenerator(
            replies=['{"documents": ["RRF 的定義是…", "RRF 的計算步驟是…"]}']
        )
        expander = LLMMultiHyDEExpander(chat_generator=generator)
        out = expander.run(queries=["RRF 是什麼?"])
        assert out["queries"] == ["RRF 是什麼?", "RRF 的定義是…", "RRF 的計算步驟是…"]

    def test_keep_original_off(self):
        generator = MockChatGenerator(replies=['{"documents": ["假設文件"]}'])
        expander = LLMMultiHyDEExpander(chat_generator=generator, keep_original=False)
        assert expander.run(queries=["原查詢"])["queries"] == ["假設文件"]

    def test_num_documents_caps_output(self):
        generator = MockChatGenerator(replies=['{"documents": ["a", "b", "c"]}'])
        expander = LLMMultiHyDEExpander(chat_generator=generator, num_documents=2)
        assert expander.run(queries=["q"])["queries"] == ["q", "a", "b"]

    def test_invalid_items_skipped(self):
        # 空字串與非字串項不算有效篇,部分成功用部分
        generator = MockChatGenerator(replies=['{"documents": ["", 42, " 有效 "]}'])
        expander = LLMMultiHyDEExpander(chat_generator=generator)
        assert expander.run(queries=["q"])["queries"] == ["q", "有效"]

    def test_garbage_reply_fails_soft(self, caplog):
        generator = MockChatGenerator(replies=["這不是 JSON"])
        expander = LLMMultiHyDEExpander(chat_generator=generator, keep_original=False)
        with caplog.at_level("WARNING"):
            out = expander.run(queries=["原查詢"])
        assert out["queries"] == ["原查詢"]
        assert "Multi-HyDE" in caplog.text

    def test_generator_exception_fails_soft(self):
        expander = LLMMultiHyDEExpander(chat_generator=BrokenGenerator())
        assert expander.run(queries=["原查詢"])["queries"] == ["原查詢"]

    def test_each_query_expanded_in_order(self):
        generator = MockChatGenerator(
            replies=['{"documents": ["假設 A"]}', '{"documents": ["假設 B"]}']
        )
        expander = LLMMultiHyDEExpander(chat_generator=generator)
        out = expander.run(queries=["qa", "qb"])
        assert out["queries"] == ["qa", "假設 A", "qb", "假設 B"]


class TestPreQRAG:
    def test_single_question_gets_rewrites(self):
        # 腳本:第 1 次呼叫 = 分類,第 2 次 = 改寫
        generator = MockChatGenerator(
            replies=[
                '{"type": "single"}',
                '{"rewrites": ["FAISS 支援的索引結構", "FAISS 索引種類"]}',
            ]
        )
        dispatcher = PreQRAGDispatcher(chat_generator=generator)
        out = dispatcher.run(queries=["FAISS 支援哪些索引?"])
        assert out["queries"] == [
            "FAISS 支援哪些索引?",
            "FAISS 支援的索引結構",
            "FAISS 索引種類",
        ]

    def test_multi_question_gets_decomposed(self):
        # 腳本:第 1 次呼叫 = 分類,第 2 次 = 拆解(行式輸出)
        generator = MockChatGenerator(
            replies=['{"type": "multi"}', "1. FAISS 的優點?\n2. ES 的優點?"]
        )
        dispatcher = PreQRAGDispatcher(chat_generator=generator)
        out = dispatcher.run(queries=["FAISS 和 ES 有什麼差別?"])
        assert out["queries"] == [
            "FAISS 和 ES 有什麼差別?",
            "FAISS 的優點?",
            "ES 的優點?",
        ]

    def test_unparsable_classification_defaults_to_single(self, caplog):
        generator = MockChatGenerator(
            replies=["這不是 JSON", '{"rewrites": ["改寫一"]}']
        )
        dispatcher = PreQRAGDispatcher(chat_generator=generator)
        with caplog.at_level("WARNING"):
            out = dispatcher.run(queries=["原查詢"])
        assert out["queries"] == ["原查詢", "改寫一"]
        assert "視為 single" in caplog.text

    def test_include_original_off(self):
        generator = MockChatGenerator(
            replies=['{"type": "single"}', '{"rewrites": ["改寫一"]}']
        )
        dispatcher = PreQRAGDispatcher(chat_generator=generator, include_original=False)
        assert dispatcher.run(queries=["原查詢"])["queries"] == ["改寫一"]

    def test_rewrite_failure_falls_back_to_original(self):
        # 分類成功但改寫壞掉:退回原查詢,且不因 include_original 而重複
        generator = MockChatGenerator(replies=['{"type": "single"}', "這不是 JSON"])
        dispatcher = PreQRAGDispatcher(chat_generator=generator)
        assert dispatcher.run(queries=["原查詢"])["queries"] == ["原查詢"]

    def test_generator_exception_fails_soft(self):
        dispatcher = PreQRAGDispatcher(chat_generator=BrokenGenerator())
        assert dispatcher.run(queries=["原查詢"])["queries"] == ["原查詢"]

    def test_num_rewrites_caps_output(self):
        generator = MockChatGenerator(
            replies=['{"type": "single"}', '{"rewrites": ["a", "b", "c"]}']
        )
        dispatcher = PreQRAGDispatcher(chat_generator=generator, num_rewrites=2)
        assert dispatcher.run(queries=["q"])["queries"] == ["q", "a", "b"]

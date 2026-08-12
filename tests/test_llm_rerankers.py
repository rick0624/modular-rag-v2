"""InsertRank 重排元件測試(離線腳本化)。"""

from __future__ import annotations

from haystack import Document

from rag.components.gateway_generator import MockChatGenerator
from rag.components.llm_rerankers import InsertRankLLMRanker


class BrokenGenerator:
    def run(self, messages):
        raise RuntimeError("LLM 掛了")


class RecordingGenerator:
    """只記錄送出的 prompt,不回答(觸發 fail-soft)。"""

    def __init__(self):
        self.prompts: list[str] = []

    def run(self, messages):
        self.prompts.append(messages[0].text)
        raise RuntimeError("只記錄,不回答")


def _docs():
    return [
        Document(id="a", content="內容 a", score=12.468),
        Document(id="b", content="內容 b", score=3.2),
        Document(id="c", content="內容 c", score=None),
    ]


class TestInsertRank:
    def test_reorders_by_scripted_ranking(self):
        generator = MockChatGenerator(replies=['{"ranking": [3, 1, 2]}'])
        ranker = InsertRankLLMRanker(chat_generator=generator, top_k=5)
        out = ranker.run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["c", "a", "b"]

    def test_scores_rendered_into_prompt(self):
        recorder = RecordingGenerator()
        ranker = InsertRankLLMRanker(chat_generator=recorder, score_label="BM25 分數")
        ranker.run(query="q", documents=_docs())
        prompt = recorder.prompts[0]
        assert "1. [BM25 分數: 12.47] 內容 a" in prompt  # 分數固定兩位小數
        assert "2. [BM25 分數: 3.20] 內容 b" in prompt
        assert "3. [BM25 分數: 無] 內容 c" in prompt  # score 為 None 渲染「無」

    def test_top_k_limits_output(self):
        generator = MockChatGenerator(replies=['{"ranking": [1, 2, 3]}'])
        ranker = InsertRankLLMRanker(chat_generator=generator, top_k=2)
        out = ranker.run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["a", "b"]

    def test_unlisted_documents_appended_in_order(self):
        generator = MockChatGenerator(replies=['{"ranking": [2]}'])
        ranker = InsertRankLLMRanker(chat_generator=generator, top_k=5)
        out = ranker.run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["b", "a", "c"]

    def test_invalid_and_duplicate_indices_ignored(self):
        generator = MockChatGenerator(replies=['{"ranking": [2, 9, 2, "x", 1, 3]}'])
        ranker = InsertRankLLMRanker(chat_generator=generator, top_k=5)
        out = ranker.run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["b", "a", "c"]

    def test_garbage_reply_fails_soft(self):
        generator = MockChatGenerator(replies=["這不是 JSON"])
        ranker = InsertRankLLMRanker(chat_generator=generator, top_k=2)
        out = ranker.run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["a", "b"]  # 原順序,仍收斂到 top_k

    def test_generator_exception_fails_soft(self):
        ranker = InsertRankLLMRanker(chat_generator=BrokenGenerator(), top_k=5)
        out = ranker.run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["a", "b", "c"]

    def test_scores_and_content_untouched(self):
        generator = MockChatGenerator(replies=['{"ranking": [2, 1, 3]}'])
        ranker = InsertRankLLMRanker(chat_generator=generator, top_k=5)
        out = ranker.run(query="q", documents=_docs())["documents"]
        assert [d.score for d in out] == [3.2, 12.468, None]  # 只重排,不改分

    def test_empty_documents(self):
        ranker = InsertRankLLMRanker(chat_generator=MockChatGenerator())
        assert ranker.run(query="q", documents=[])["documents"] == []

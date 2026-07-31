"""重排測試:LLMRanker 搭配 mock chat generator(離線腳本化)。"""

from __future__ import annotations

from haystack import Document
from haystack.components.rankers import LLMRanker

from rag.components.gateway_generator import MockChatGenerator


def _docs():
    return [
        Document(id="a::chunk_0", content="低相關", meta={"chunk_id": "a::chunk_0"}),
        Document(id="b::chunk_0", content="高相關", meta={"chunk_id": "b::chunk_0"}),
        Document(id="c::chunk_0", content="中相關", meta={"chunk_id": "c::chunk_0"}),
    ]


def test_llm_ranker_reorders_by_scripted_indices():
    # LLMRanker 的協定:回傳 JSON,index 為文件在輸入清單中的 1-based 序號
    generator = MockChatGenerator(
        replies=['{"documents": [{"index": 2}, {"index": 3}]}']
    )
    ranker = LLMRanker(chat_generator=generator, top_k=5)
    out = ranker.run(query="哪個高相關?", documents=_docs())["documents"]
    assert [d.id for d in out] == ["b::chunk_0", "c::chunk_0"], (
        "應依 LLM 回傳順序重排,且未列出的文件被過濾"
    )


def test_llm_ranker_top_k_limits_output():
    generator = MockChatGenerator(
        replies=['{"documents": [{"index": 1}, {"index": 2}, {"index": 3}]}']
    )
    ranker = LLMRanker(chat_generator=generator, top_k=2)
    out = ranker.run(query="q", documents=_docs())["documents"]
    assert len(out) == 2


def test_llm_ranker_garbled_reply_fails_soft():
    # raise_on_failure 預設 False:解析失敗時回傳原始順序,查詢不中斷
    generator = MockChatGenerator(replies=["這不是 JSON"])
    ranker = LLMRanker(chat_generator=generator, top_k=5)
    out = ranker.run(query="q", documents=_docs())["documents"]
    assert [d.id for d in out] == ["a::chunk_0", "b::chunk_0", "c::chunk_0"]

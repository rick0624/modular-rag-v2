"""api_rerank 測試:請求體組裝、回應形狀對映、index 語意、fail-soft。"""

from __future__ import annotations

import logging

import pytest
from conftest import FakeClient, FakeResponse

from haystack import Document

from rag.components.api_clients import FlexibleAPIRanker
from rag.errors import APICallError, APIResponseFormatError

ENDPOINT = "https://rerank.example.com/v1/rerank"


def _docs():
    return [
        Document(id="a::chunk_0", content="第一筆", meta={"chunk_id": "a::chunk_0"}, score=0.1),
        Document(id="b::chunk_0", content="第二筆", meta={"chunk_id": "b::chunk_0"}, score=0.2),
        Document(id="c::chunk_0", content="第三筆", meta={"chunk_id": "c::chunk_0"}, score=0.3),
    ]


def _ranker(client, **kwargs):
    return FlexibleAPIRanker(client=client, endpoint=ENDPOINT, **kwargs)


def _payload(*pairs):
    return {"returnData": [{"index": i, "score": s} for i, s in pairs]}


class TestRequestBody:
    def test_default_fields_match_company_api(self):
        client = FakeClient([FakeResponse(payload=_payload((0, 0.9)))])
        _ranker(client, model="rerank-v1").run(query="問題", documents=_docs())
        assert client.calls[0]["json"] == {
            "question": "問題",
            "documents": ["第一筆", "第二筆", "第三筆"],
            "model": "rerank-v1",
        }
        assert client.calls[0]["url"] == ENDPOINT

    def test_model_omitted_when_none(self):
        client = FakeClient([FakeResponse(payload=_payload((0, 0.9)))])
        _ranker(client).run(query="q", documents=_docs())
        assert "model" not in client.calls[0]["json"]

    def test_field_names_are_configurable(self):
        client = FakeClient([FakeResponse(payload={"results": [{"i": 0, "s": 1.0}]})])
        ranker = _ranker(
            client,
            query_field="query",
            documents_field="passages",
            results_field="results",
            index_field="i",
            score_field="s",
        )
        out = ranker.run(query="q", documents=_docs())["documents"]
        assert set(client.calls[0]["json"]) == {"query", "passages"}
        assert [d.id for d in out] == ["a::chunk_0"]

    def test_headers_forwarded(self):
        client = FakeClient([FakeResponse(payload=_payload((0, 0.9)))])
        _ranker(client, headers={"Authorization": "Bearer T"}).run(
            query="q", documents=_docs()
        )
        assert client.calls[0]["headers"]["Authorization"] == "Bearer T"

    def test_empty_documents_skips_the_call(self):
        client = FakeClient([])
        assert _ranker(client).run(query="q", documents=[])["documents"] == []
        assert client.calls == []


class TestOrdering:
    def test_reorders_by_score_not_by_response_order(self):
        # API 未必已排序:分數才是準,回應順序不是
        client = FakeClient([FakeResponse(payload=_payload((0, 0.1), (1, 0.9), (2, 0.5)))])
        out = _ranker(client).run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["b::chunk_0", "c::chunk_0", "a::chunk_0"]

    def test_score_written_back_to_documents(self):
        client = FakeClient([FakeResponse(payload=_payload((1, 0.9)))])
        out = _ranker(client).run(query="q", documents=_docs())["documents"]
        assert out[0].score == 0.9  # 覆寫檢索分數,供下游 max_score 融合使用

    def test_higher_is_better_false_flips_order(self):
        # 回傳距離(越小越相關)的 API
        client = FakeClient([FakeResponse(payload=_payload((0, 0.8), (1, 0.2)))])
        out = _ranker(client, higher_is_better=False).run(query="q", documents=_docs())
        assert [d.id for d in out["documents"]] == ["b::chunk_0", "a::chunk_0"]

    def test_top_k_truncates_after_sorting(self):
        client = FakeClient([FakeResponse(payload=_payload((0, 0.1), (1, 0.9), (2, 0.5)))])
        out = _ranker(client, top_k=2).run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["b::chunk_0", "c::chunk_0"]

    def test_ties_fall_back_to_input_order(self):
        # 同分時不能取決於 API 回應的排列,否則同一組輸入會有不同輸出
        client = FakeClient([FakeResponse(payload=_payload((2, 0.5), (0, 0.5), (1, 0.5)))])
        out = _ranker(client).run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["a::chunk_0", "b::chunk_0", "c::chunk_0"]

    def test_metadata_and_id_survive_rescoring(self):
        client = FakeClient([FakeResponse(payload=_payload((1, 0.9)))])
        out = _ranker(client).run(query="q", documents=_docs())["documents"]
        assert out[0].id == "b::chunk_0"
        assert out[0].meta == {"chunk_id": "b::chunk_0"}
        assert out[0].content == "第二筆"

    def test_unlisted_documents_are_dropped(self):
        client = FakeClient([FakeResponse(payload=_payload((2, 0.7)))])
        out = _ranker(client).run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["c::chunk_0"]


class TestIndexSemantics:
    def test_index_base_one_shifts_positions(self):
        client = FakeClient([FakeResponse(payload=_payload((1, 0.9), (2, 0.5)))])
        out = _ranker(client, index_base=1).run(query="q", documents=_docs())
        assert [d.id for d in out["documents"]] == ["a::chunk_0", "b::chunk_0"]

    def test_all_out_of_range_hints_at_index_base(self):
        # 1 起算的 API 配 index_base: 0 → 三筆送出、收到 index 3 越界
        client = FakeClient([FakeResponse(payload=_payload((3, 0.9)))])
        with pytest.raises(APIResponseFormatError, match="index_base: 1"):
            _ranker(client, raise_on_failure=True).run(query="q", documents=_docs())

    def test_partial_out_of_range_is_ignored_with_warning(self, caplog):
        client = FakeClient([FakeResponse(payload=_payload((0, 0.9), (99, 0.5)))])
        with caplog.at_level(logging.WARNING):
            out = _ranker(client).run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["a::chunk_0"]
        assert "越界" in caplog.text

    def test_duplicate_index_keeps_first(self, caplog):
        client = FakeClient([FakeResponse(payload=_payload((0, 0.9), (0, 0.1)))])
        with caplog.at_level(logging.WARNING):
            out = _ranker(client).run(query="q", documents=_docs())["documents"]
        assert [(d.id, d.score) for d in out] == [("a::chunk_0", 0.9)]
        assert "重複" in caplog.text


class TestResponseShapes:
    def test_bare_list_response(self):
        client = FakeClient([FakeResponse(payload=[{"index": 1, "score": 0.9}])])
        out = _ranker(client, results_field=None).run(query="q", documents=_docs())
        assert [d.id for d in out["documents"]] == ["b::chunk_0"]

    def test_nested_dot_path(self):
        client = FakeClient(
            [FakeResponse(payload={"result": {"items": [{"index": 2, "score": 1.0}]}})]
        )
        out = _ranker(client, results_field="result.items").run(
            query="q", documents=_docs()
        )
        assert [d.id for d in out["documents"]] == ["c::chunk_0"]

    def test_missing_results_field_lists_actual_keys(self):
        client = FakeClient([FakeResponse(payload={"data": []})])
        with pytest.raises(APIResponseFormatError, match=r"找不到 'returnData'.*\['data'\]"):
            _ranker(client, raise_on_failure=True).run(query="q", documents=_docs())

    def test_missing_score_field_lists_actual_keys(self):
        client = FakeClient([FakeResponse(payload={"returnData": [{"index": 0}]})])
        with pytest.raises(APIResponseFormatError, match="缺少 'score'"):
            _ranker(client, raise_on_failure=True).run(query="q", documents=_docs())

    def test_non_numeric_score_reports_the_value(self):
        client = FakeClient(
            [FakeResponse(payload={"returnData": [{"index": 0, "score": "高"}]})]
        )
        with pytest.raises(APIResponseFormatError, match="不是數字"):
            _ranker(client, raise_on_failure=True).run(query="q", documents=_docs())


class TestFailSoft:
    def test_api_error_keeps_retrieval_order(self, caplog):
        client = FakeClient([FakeResponse(status_code=500, text="boom")])
        with caplog.at_level(logging.WARNING):
            out = _ranker(client).run(query="q", documents=_docs())["documents"]
        assert [d.id for d in out] == ["a::chunk_0", "b::chunk_0", "c::chunk_0"]
        assert "rerank API 失敗" in caplog.text

    def test_fail_soft_still_respects_top_k(self):
        client = FakeClient([FakeResponse(status_code=500, text="boom")])
        out = _ranker(client, top_k=2).run(query="q", documents=_docs())["documents"]
        assert len(out) == 2

    def test_raise_on_failure_propagates(self):
        client = FakeClient([FakeResponse(status_code=401, text="unauthorized")])
        with pytest.raises(APICallError, match="401"):
            _ranker(client, raise_on_failure=True).run(query="q", documents=_docs())

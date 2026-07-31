"""SubqueryFusion 測試:v1 fuse() 語意的數值案例移植。"""

from __future__ import annotations

import pytest

from haystack import Document

from rag.components.fusion import SubqueryFusion
from rag.errors import ComponentError


def doc(chunk_id: str, score: float, *, doc_id: str | None = None, page: int | None = None):
    did = doc_id or chunk_id.split("::")[0]
    return Document(
        id=chunk_id,
        content=f"內容 {chunk_id}",
        score=score,
        meta={"chunk_id": chunk_id, "doc_id": did, "page": page, "seq": 0},
    )


class TestActivationRule:
    def test_single_source_without_fusion_config_passes_through(self):
        docs = [doc("a::chunk_0", 0.9), doc("b::chunk_0", 0.5)]
        out = SubqueryFusion(always_fuse=False).run(results=[docs])["documents"]
        assert out == docs  # 原樣通過,連 top_k 都不套用

    def test_single_source_with_fusion_config_still_aggregates(self):
        docs = [doc("a::chunk_0", 0.9), doc("a::chunk_1", 0.5)]
        out = SubqueryFusion(group_by="doc", always_fuse=True).run(results=[docs])[
            "documents"
        ]
        assert len(out) == 1
        assert out[0].meta["group_key"] == "a"
        assert out[0].meta["num_merged"] == 2

    def test_empty_results_raise(self):
        with pytest.raises(ComponentError, match="沒有任何檢索結果可融合"):
            SubqueryFusion().run(results=[])


class TestStrategies:
    def _two_sources(self):
        # chunk c 同時出現在兩路(各排第 2),a/b 各只出現一次(排第 1)
        source_1 = [doc("a::chunk_0", 0.9), doc("c::chunk_0", 0.8)]
        source_2 = [doc("b::chunk_0", 0.7), doc("c::chunk_0", 0.6)]
        return [source_1, source_2]

    def test_rrf_boosts_common_chunk(self):
        out = SubqueryFusion(strategy="rrf", top_k=3).run(results=self._two_sources())[
            "documents"
        ]
        assert out[0].meta["chunk_id"] == "c::chunk_0", "兩路都出現的切片 RRF 應最高"
        assert out[0].score == pytest.approx(2 / 62)
        assert out[1].score == pytest.approx(1 / 61)
        # 代表切片取最高原始分數的那份;sources 記錄各路名次與原始分數
        assert out[0].meta["num_merged"] == 2
        assert out[0].meta["sources"] == [
            {"subquery_index": 0, "rank": 2, "score": 0.8},
            {"subquery_index": 1, "rank": 2, "score": 0.6},
        ]

    def test_max_score_uses_best_original_score(self):
        out = SubqueryFusion(strategy="max_score", top_k=3).run(
            results=self._two_sources()
        )["documents"]
        assert [d.meta["chunk_id"] for d in out] == [
            "a::chunk_0",
            "c::chunk_0",
            "b::chunk_0",
        ]
        assert out[0].score == pytest.approx(0.9)
        assert out[1].score == pytest.approx(0.8), "c 取兩路中較高的原始分"

    def test_concat_dedup_preserves_source_order_with_synthetic_scores(self):
        out = SubqueryFusion(strategy="concat_dedup", top_k=4).run(
            results=self._two_sources()
        )["documents"]
        assert [d.meta["chunk_id"] for d in out] == [
            "a::chunk_0",
            "c::chunk_0",
            "b::chunk_0",
        ]
        # 名次型合成分數:1/(first_order+1),維持降冪不變量
        assert [d.score for d in out] == pytest.approx([1.0, 0.5, 1 / 3])

    def test_top_k_cuts_after_ranking(self):
        out = SubqueryFusion(strategy="rrf", top_k=1).run(results=self._two_sources())[
            "documents"
        ]
        assert len(out) == 1


class TestGroupBy:
    def test_group_by_doc_merges_chunks_of_same_document(self):
        source = [
            doc("d1::chunk_0", 0.9),
            doc("d1::chunk_1", 0.8),
            doc("d2::chunk_0", 0.7),
        ]
        out = SubqueryFusion(group_by="doc", always_fuse=True, top_k=5).run(
            results=[source]
        )["documents"]
        keys = [d.meta["group_key"] for d in out]
        assert keys == ["d1", "d2"]
        best = out[0]
        assert best.meta["chunk_id"] == "d1::chunk_0", "代表切片 = 組內最高原始分"
        assert best.meta["num_merged"] == 2

    def test_group_by_page_key_format(self):
        source = [
            doc("d1::chunk_0", 0.9, page=2),
            doc("d1::chunk_1", 0.8, page=2),
            doc("d1::chunk_2", 0.7, page=3),
        ]
        out = SubqueryFusion(group_by="page", always_fuse=True).run(results=[source])[
            "documents"
        ]
        assert [d.meta["group_key"] for d in out] == ["d1#p2", "d1#p3"]

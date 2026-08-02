"""融合 / 聚合元件:pipeline 的內建步驟(v1 ``fuse()`` 的精確移植)。

把 N 個(重排後的)子查詢檢索結果合併成單一文件清單:

- 多個 subquery 的結果合併(query decomposition 檢索);
- 依 ``group_by`` 把切片聚合成 doc / page 粒度(每組取最高分代表切片)。

排序策略:

- ``rrf``(預設):Reciprocal Rank Fusion,以各來源內的**名次**計分
  (score = Σ 1/(60+rank))— 跨 subquery 的原始分數不可直接比較,
  名次法是安全預設。
- ``concat_dedup``:依「來源順序 → 名次」保序去重;輸出分數為名次型
  合成值(維持降冪不變量),原始分數保留在 metadata 的 sources。
- ``max_score``:各組取最高原始分數排序 — 僅適用於各來源分數同量綱
  (同一個 reranker)的場合。

聚合後每組的代表切片 meta 記 ``group_key`` / ``num_merged`` /
``sources``(各來源的 subquery_index / rank / score)。

啟用規則(v1 語意):子查詢數 > 1,或 config 有 ``inference.fusion``
區塊時才融合;單一查詢且未設定 fusion 時原樣通過(傳統直線流程)。
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

from haystack import Document, component

from rag.errors import ComponentError

GroupBy = Literal["none", "doc", "page"]
Strategy = Literal["rrf", "concat_dedup", "max_score"]

_RRF_K = 60


def _group_key(doc: Document, group_by: GroupBy) -> str:
    doc_id = doc.meta.get("doc_id") or doc.id
    if group_by == "doc":
        return doc_id
    if group_by == "page":
        return f"{doc_id}#p{doc.meta.get('page')}"
    return doc.meta.get("chunk_id") or doc.id


def _score(doc: Document) -> float:
    return doc.score if doc.score is not None else 0.0


@component
class SubqueryFusion:
    """把 N 路檢索結果融合 / 聚合成單一文件清單。"""

    def __init__(
        self,
        group_by: GroupBy = "none",
        strategy: Strategy = "rrf",
        top_k: int = 5,
        always_fuse: bool = False,
    ) -> None:
        self.group_by = group_by
        self.strategy = strategy
        self.top_k = top_k
        # config 有 fusion 區塊時為 True:單一查詢也走融合(可做 doc/page 聚合)。
        self.always_fuse = always_fuse

    @component.output_types(documents=list[Document], applied=bool)
    def run(self, results: list[list[Document]]) -> dict[str, Any]:
        """融合 N 路結果(N ≥ 1);單路且未設定 fusion 時原樣通過。

        ``applied`` 指出這次是否真的做了融合 —— 元件恆存在於 pipeline
        中,穿透時也會出現在 trace,靠這個旗標才分得出「融合過」與
        「只是路過」,否則紀錄會讓人誤以為做了去重 / 聚合 / 排序。

        Raises:
            ComponentError: ``results`` 為空(上游沒有任何檢索結果)。
        """
        if not results:
            raise ComponentError("沒有任何檢索結果可融合")
        if len(results) == 1 and not self.always_fuse:
            return {"documents": results[0], "applied": False}

        groups: dict[str, dict[str, Any]] = {}
        order = 0
        for source_index, documents in enumerate(results):
            for rank, doc in enumerate(documents, start=1):
                key = _group_key(doc, self.group_by)
                group = groups.get(key)
                if group is None:
                    group = {"best": doc, "rrf": 0.0, "first_order": order, "sources": []}
                    groups[key] = group
                    order += 1
                group["rrf"] += 1.0 / (_RRF_K + rank)
                group["sources"].append(
                    {"subquery_index": source_index, "rank": rank, "score": _score(doc)}
                )
                if _score(doc) > _score(group["best"]):
                    group["best"] = doc

        if self.strategy == "rrf":
            ranked = sorted(groups.items(), key=lambda kv: kv[1]["rrf"], reverse=True)
            scores = {key: group["rrf"] for key, group in groups.items()}
        elif self.strategy == "max_score":
            ranked = sorted(
                groups.items(), key=lambda kv: _score(kv[1]["best"]), reverse=True
            )
            scores = {key: _score(group["best"]) for key, group in groups.items()}
        else:  # concat_dedup:保序去重;分數用名次型合成值以維持降冪不變量。
            ranked = sorted(groups.items(), key=lambda kv: kv[1]["first_order"])
            scores = {
                key: 1.0 / (group["first_order"] + 1) for key, group in groups.items()
            }

        fused = [
            dataclasses.replace(
                group["best"],
                score=scores[key],
                meta={
                    **group["best"].meta,
                    "group_key": key,
                    "num_merged": len(group["sources"]),
                    "sources": group["sources"],
                },
            )
            for key, group in ranked[: self.top_k]
        ]
        return {"documents": fused, "applied": True}

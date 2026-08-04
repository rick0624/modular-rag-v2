"""公司 fusion custom module 骨架(佔位範例,邏輯待替換)。

``fusion`` 槽位的 custom module 範本:把 N 條子查詢各自(已重排的)檢索
結果合併成單一清單。內建融合(rrf / concat_dedup / max_score ×
group_by)不夠用 —— 例如要按部門加權、同一份文件只留最新版本、或打
公司 API 查權限過濾 —— 時就寫這一支。

槽位契約:``results: list[list[Document]] → documents: list[Document]``。

三條紀律:

1. **掛上即一律執行**:單一查詢(``len(results) == 1``)也會進來,
   原樣通過與否由你決定 —— 內建融合的「單查詢穿透」不會幫你做。
2. **跨子查詢的原始分數不可直接比較**(不同子查詢、不同候選池),
   安全預設是名次法(RRF,本範例採用);要用原始分數請先確認各路
   同量綱(同一個 reranker)。
3. 建議額外輸出 ``applied: bool``(額外輸出自動進 trace):trace 靠它
   分辨「融合過」與「原樣通過」,不回報時 log 會顯示「元件未回報」。

掛載方式(configs/custom_demo.yaml):

    fusion:
      method: custom
      params:
        file: examples/custom_modules/company_fusion.py
        class: CompanyFusion
        init_params:
          top_k: 3
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from haystack import component
from haystack.dataclasses import Document

# logger 命名在 "rag.*" 底下:file: 與 class_path: 兩種載入方式都吃得到
# 框架的 log 檔設定(見 README「自訂方法」的「log 要看得到」)。
logger = logging.getLogger("rag.custom.company_fusion")

_RRF_K = 60  # RRF 的名次平滑常數(業界慣用 60)


@component
class CompanyFusion:
    """RRF 名次融合 + chunk_id 去重(公司邏輯的替換起點)。

    Args:
        top_k: 融合後保留的筆數上限。搭配下游 prompt 使用時別設太大
            (融合輸出直接進 prompt)。
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    @component.output_types(documents=list[Document], applied=bool)
    def run(self, results: list[list[Document]]) -> dict[str, Any]:
        """融合 N 路結果,回傳 ``{"documents": [...], "applied": True}``。

        TODO(替換點):把下面的 RRF 換成公司的融合邏輯,例如::

            # 同一份文件(dockey)只留版本最新的切片,再按部門加權排序
            latest = keep_latest_version_per_dockey(results)
            return {"documents": rank_by_department_weight(latest)[: self.top_k],
                    "applied": True}

        紀律:輸出必須按分數**降冪**(下游顯示假設如此),meta 請從
        原始 Document 繼承(doc_id / chunk_id 是 trace 與 evaluation 的
        依據),不要回空清單以外的驚喜。
        """
        groups: dict[str, dict[str, Any]] = {}
        for documents in results:
            for rank, doc in enumerate(documents, start=1):
                key = doc.meta.get("chunk_id") or doc.id
                group = groups.setdefault(key, {"best": doc, "rrf": 0.0, "hits": 0})
                group["rrf"] += 1.0 / (_RRF_K + rank)
                group["hits"] += 1
                best_score = group["best"].score or 0.0
                if (doc.score or 0.0) > best_score:
                    group["best"] = doc
        ranked = sorted(groups.values(), key=lambda g: g["rrf"], reverse=True)
        fused = [
            dataclasses.replace(
                group["best"],
                score=group["rrf"],
                meta={**group["best"].meta, "num_merged": group["hits"]},
            )
            for group in ranked[: self.top_k]
        ]
        logger.debug(
            "CompanyFusion:%d 路 → %d 筆(top_k=%d)",
            len(results), len(fused), self.top_k,
        )
        return {"documents": fused, "applied": True}

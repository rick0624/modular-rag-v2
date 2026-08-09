"""Inference 的兩條選填支線元件:routing(查詢分類)與 formatter(對外格式)。

兩者都不影響檢索:

- :class:`KeywordRouteClassifier`(``routing: keyword_match``):獨立支線,
  吃「原始」查詢(不經 transform 鏈)、輸出 ``route: dict``、圖上沒有
  下游 —— 結果由 ``query()`` 收進回傳值的 ``routing`` 鍵與 trace。
- :class:`SimpleJsonFormatter`(``formatter: simple_json``):終端支線
  (fusion 之後,與 prompt → generation 並聯),把融合後的文件組成
  **對外**格式,進 ``query()`` 回傳值的 ``output`` 鍵;canonical 鍵照舊。

正式場景(分類模型、公司信封)請用 ``method: custom`` 接自訂元件,
契約見 README「如何新增一個自訂方法」節。
"""

from __future__ import annotations

import logging
from typing import Any

from haystack import Document, component

from rag.errors import ComponentError

logger = logging.getLogger(__name__)


@component
class KeywordRouteClassifier:
    """關鍵字規則分類:逐類別計算命中數,取命中率最高者。

    Args:
        routes: 類別 → 關鍵字清單。查詢文字包含關鍵字即計一次命中。
        default_category: 沒有任何關鍵字命中時回傳的類別。
    """

    def __init__(
        self, routes: dict[str, list[str]], default_category: str = "general"
    ) -> None:
        if not routes:
            raise ComponentError(
                "keyword_match routing 需要至少一個類別:"
                "請在 params.routes 提供 {類別: [關鍵字, ...]}"
            )
        for category, keywords in routes.items():
            if not keywords:
                raise ComponentError(
                    f"routing 類別 '{category}' 的關鍵字清單是空的;"
                    "每個類別至少需要一個關鍵字"
                )
        self.routes = {
            category: [str(kw) for kw in keywords]
            for category, keywords in routes.items()
        }
        self.default_category = default_category

    @component.output_types(route=dict[str, Any])
    def run(self, query: str) -> dict[str, Any]:
        """回傳 ``{"route": {category, confidence, matched_keywords}}``。

        confidence = 勝出類別的命中關鍵字數 / 該類別關鍵字總數,
        僅供相對參考(規則分類沒有機率意義)。
        """
        best_category = self.default_category
        best_matched: list[str] = []
        best_confidence = 0.0
        for category, keywords in self.routes.items():
            matched = [kw for kw in keywords if kw in query]
            confidence = len(matched) / len(keywords)
            if len(matched) > len(best_matched) or (
                len(matched) == len(best_matched)
                and matched
                and confidence > best_confidence
            ):
                best_category = category
                best_matched = matched
                best_confidence = confidence
        return {
            "route": {
                "category": best_category,
                "confidence": round(best_confidence, 4),
                "matched_keywords": best_matched,
            }
        }


@component
class SimpleJsonFormatter:
    """把融合後的文件組成通用 JSON 形狀。

    輸出形狀::

        {
          "query": 原始查詢,
          "total": 筆數,
          "documents": [
            {"doc_id", "chunk_id", "page", "score", "content"(選配)}, ...
          ],
        }

    Args:
        include_content: 是否包含切片內文(只要引用資訊時設 false,
            回應可以小很多)。
    """

    def __init__(self, include_content: bool = True) -> None:
        self.include_content = include_content

    @component.output_types(payload=dict[str, Any])
    def run(self, documents: list[Document], query: str) -> dict[str, Any]:
        rows = []
        for doc in documents:
            row: dict[str, Any] = {
                "doc_id": doc.meta.get("doc_id"),
                "chunk_id": doc.meta.get("chunk_id"),
                "page": doc.meta.get("page"),
                "score": doc.score,
            }
            if self.include_content:
                row["content"] = doc.content
            rows.append(row)
        return {
            "payload": {"query": query, "total": len(rows), "documents": rows}
        }

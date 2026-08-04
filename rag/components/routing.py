"""Question routing 元件:判斷查詢類別,結果附加於輸出。

routing 是 inference 的獨立支線:吃「原始」查詢(不經 transform 鏈)、
輸出 ``route: dict``、圖上沒有下游 —— 判斷結果不影響檢索行為,
由 ``RagPipelines.query()`` 收進回傳值的 ``routing`` key 與 trace。

內建的 :class:`KeywordRouteClassifier` 是展示/測試用的簡單規則分類;
正式場景(domain knowhow 的分類模型、規則引擎)請用 ``method: custom``
接自訂元件,契約同樣是 ``query: str → route: dict[str, Any]``。
"""

from __future__ import annotations

import logging
from typing import Any

from haystack import component

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

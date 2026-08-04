"""公司 question routing custom module 骨架(佔位範例,邏輯待替換)。

``routing`` 槽位的 custom module 範本:判斷查詢屬於哪個類別。
routing 是獨立支線 —— 吃「原始」查詢、結果附加在 ``query()`` 回傳的
``routing`` key(與 trace),完全不影響檢索行為。

槽位契約:``query: str → route: dict[str, Any]``。route dict 的內容
自由(慣例上含 ``category``);框架原樣往外帶,公司系統拿到後自行運用。

掛載方式(configs/custom_demo.yaml):

    routing:
      method: custom
      params:
        file: examples/custom_modules/company_router.py
        class: CompanyQuestionRouter
"""

from __future__ import annotations

import logging
from typing import Any

from haystack import component

# logger 命名在 "rag.*" 底下:file: 與 class_path: 兩種載入方式都吃得到
# 框架的 log 檔設定(見 README「自訂方法」的「log 要看得到」)。
logger = logging.getLogger("rag.custom.company_router")

# 示範規則(TODO(替換點):接上真正的分類邏輯後刪除)。
_DEFAULT_RULES: dict[str, list[str]] = {
    "人資/差勤": ["請假", "特休", "差勤", "薪資", "加班"],
    "資訊系統": ["VPN", "帳號", "密碼", "系統", "登入"],
}


@component
class CompanyQuestionRouter:
    """依 domain knowhow 判斷查詢類別。

    Args:
        rules: 類別 → 關鍵字清單;None 時使用內建示範規則。
        default_category: 無任何規則命中時回傳的類別。
    """

    def __init__(
        self,
        rules: dict[str, list[str]] | None = None,
        default_category: str = "其他",
    ) -> None:
        self.rules = rules or _DEFAULT_RULES
        self.default_category = default_category

    @component.output_types(route=dict[str, Any])
    def run(self, query: str) -> dict[str, Any]:
        """判斷 ``query`` 的類別,回傳 ``{"route": {...}}``。

        TODO(替換點):把下面的關鍵字規則換成公司的分類模型 /
        規則引擎呼叫,例如::

            response = httpx.post(self.endpoint, json={"question": query})
            verdict = response.json()          # 公司回傳的判斷結果
            return {"route": {"category": verdict["label"], **verdict}}

        route dict 的形狀由你決定(框架原樣往外帶),
        慣例上至少放 ``category``。
        """
        best_category = self.default_category
        best_matched: list[str] = []
        for category, keywords in self.rules.items():
            matched = [kw for kw in keywords if kw in query]
            if len(matched) > len(best_matched):
                best_category = category
                best_matched = matched
        route = {
            "category": best_category,
            "confidence": (
                round(len(best_matched) / len(self.rules[best_category]), 4)
                if best_matched
                else 0.0
            ),
            "matched_keywords": best_matched,
        }
        logger.debug("CompanyQuestionRouter:%r → %s", query, route)
        return {"route": route}

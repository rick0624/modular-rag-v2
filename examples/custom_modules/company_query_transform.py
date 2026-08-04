"""公司 query transformation custom module 骨架(佔位範例,邏輯待替換)。

``query_transformation`` 槽位的 custom module 範本:把使用者的原始問句
改寫成更適合公司語料的查詢。這裡示範最常見的兩件事 —— 補上部門黑話的
標準名稱、把一句話拆成多條子查詢(下游會逐條檢索,再由 fusion 合併)。

槽位契約:``queries: list[str] → queries: list[str]``。**進出都是清單**
(不是單一字串):上游可能已經拆過,拆解型的元件也要能再拆。

掛載方式(configs/custom_demo.yaml):

    query_transformation:
      method: [normalize, custom]     # custom 可放在方法鏈中
      method_params:
        custom:
          file: examples/custom_modules/company_query_transform.py
          class: CompanyQueryExpander
"""

from __future__ import annotations

import logging
from typing import Any

from haystack import component

# logger 命名在 "rag.*" 底下:file: 與 class_path: 兩種載入方式都吃得到
# 框架的 log 檔設定(見 README「自訂方法」的「log 要看得到」)。
logger = logging.getLogger("rag.custom.query_expander")

# 示範用的公司黑話對照(TODO(替換點):接上真正的詞庫後刪除)。
_DEFAULT_SYNONYMS: dict[str, str] = {
    "請假": "請假 差勤申請",
    "報帳": "報帳 費用核銷",
    "VPN": "VPN 遠端連線",
}


@component
class CompanyQueryExpander:
    """以公司詞庫擴充查詢,必要時再拆成多條子查詢。

    Args:
        synonyms: 詞 → 擴充後字串;None 時使用內建示範詞庫。
        split_on: 命中這些連接詞就拆成多條子查詢(空清單 = 不拆)。
        max_queries: 輸出的查詢數上限(防止拆解爆炸;每多一條就多一輪檢索)。
    """

    def __init__(
        self,
        synonyms: dict[str, str] | None = None,
        split_on: list[str] | None = None,
        max_queries: int = 4,
    ) -> None:
        self.synonyms = synonyms if synonyms is not None else _DEFAULT_SYNONYMS
        self.split_on = split_on if split_on is not None else ["以及", "還有", ";"]
        self.max_queries = max_queries

    @component.output_types(queries=list[str])
    def run(self, queries: list[str]) -> dict[str, Any]:
        """改寫 ``queries``,回傳 ``{"queries": [...]}``。

        TODO(替換點):把下面的字串處理換成公司的改寫服務,例如::

            response = httpx.post(self.endpoint, json={"questions": queries})
            return {"queries": response.json()["rewritten"]}

        兩條紀律:輸出**必須**是 ``list[str]``(即使只有一條),且不要回
        空清單 —— 下游沒有查詢可跑。改寫失敗時原樣傳回輸入最安全。
        """
        expanded: list[str] = []
        for query in queries:
            for part in self._split(query):
                rewritten = self._apply_synonyms(part)
                if rewritten not in expanded:
                    expanded.append(rewritten)
        result = expanded[: self.max_queries] or list(queries)
        logger.debug("CompanyQueryExpander:%s → %s", queries, result)
        return {"queries": result}

    def _split(self, query: str) -> list[str]:
        """依連接詞拆成多條(沒命中就是原本那一條)。"""
        parts = [query]
        for token in self.split_on:
            parts = [piece for part in parts for piece in part.split(token)]
        return [part.strip() for part in parts if part.strip()] or [query]

    def _apply_synonyms(self, query: str) -> str:
        """命中詞庫就把標準名稱補進查詢(原詞保留,不做破壞性替換)。"""
        for term, expansion in self.synonyms.items():
            if term in query and expansion not in query:
                query = query.replace(term, expansion)
        return query

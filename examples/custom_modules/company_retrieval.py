"""公司檢索 custom module 骨架(佔位範例,邏輯待替換)。

這是 ``retrieval`` 槽位的 custom module 範本:示範「呼叫公司檢索服務 →
把公司欄位映射成框架的 canonical 型別 ``Document``」。目前
``_call_search_api`` 回傳寫死的模擬資料;到公司後只要把該函式換成
真正的 retrieve_service 呼叫(query embedding + POST ES search),
其餘映射邏輯照用即可。

槽位契約:``query: str → documents: list[Document]``(建構期會驗證)。

掛載方式(configs/custom_demo.yaml):

    retrieval:
      method: custom
      params:
        file: examples/custom_modules/company_retrieval.py
        class: CompanyRetriever
        init_params: {top_k: 5}
"""

from __future__ import annotations

import logging
from typing import Any

from haystack import component
from haystack.dataclasses import Document

# logger 命名在 "rag.*" 底下:file: 與 class_path: 兩種載入方式都吃得到
# 框架的 log 檔設定(見 README「自訂方法」的「log 要看得到」)。
logger = logging.getLogger("rag.custom.company_retrieval")

# 模擬的公司知識庫(TODO(替換點):接上真正的服務後整段刪除)。
_FAKE_CORPUS: list[dict[str, Any]] = [
    {
        "dockey": "HR-0001",
        "score": 0.92,
        "contentTitle": "請假規則",
        "contentChunk": "特休假依年資核給,請假需於三日前於差勤系統提出申請。",
    },
    {
        "dockey": "HR-0002",
        "score": 0.85,
        "contentTitle": "薪資發放",
        "contentChunk": "薪資於每月五日發放,遇假日提前至前一個工作日。",
    },
    {
        "dockey": "IT-0001",
        "score": 0.78,
        "contentTitle": "VPN 連線設定",
        "contentChunk": "遠端工作需先安裝公司 VPN,帳號密碼與網域登入相同。",
    },
    {
        "dockey": "IT-0002",
        "score": 0.66,
        "contentTitle": "密碼重設",
        "contentChunk": "密碼每九十日需變更一次,忘記密碼請聯絡資訊服務台。",
    },
]


@component
class CompanyRetriever:
    """呼叫公司檢索服務,把回應映射成 ``list[Document]``。

    Args:
        endpoint: 公司檢索 API 端點(佔位參數,示範 init_params 傳遞)。
        top_k: 取回筆數上限。
    """

    def __init__(
        self,
        endpoint: str = "https://km.example.com/api/search",
        top_k: int = 5,
    ) -> None:
        self.endpoint = endpoint
        self.top_k = top_k

    def _call_search_api(self, query: str) -> list[dict[str, Any]]:
        """呼叫公司檢索服務,回傳原始 hit 清單。

        TODO(替換點):這裡換成真正的 retrieve_service 邏輯 ——
        query embedding + POST 呼叫 ES search,例如::

            response = httpx.post(
                self.endpoint,
                json={"question": query, "size": self.top_k},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()["hits"]

        目前的佔位實作:對模擬語料做簡單的字元重疊過濾,
        讓範例離線可跑、且不同查詢會得到不同結果。
        """
        query_chars = set(query)
        hits = [
            hit
            for hit in _FAKE_CORPUS
            if query_chars & set(hit["contentTitle"] + hit["contentChunk"])
        ]
        return hits or list(_FAKE_CORPUS)  # 全無交集時退回整批(展示用)

    @component.output_types(documents=list[Document])
    def run(self, query: str) -> dict[str, Any]:
        hits = self._call_search_api(query)[: self.top_k]
        # 欄位映射(本範例的核心):公司欄位 → 框架 canonical 型別。
        #   contentChunk → Document.content(下游 rerank / prompt 讀這裡)
        #   score        → Document.score
        #   其餘欄位      → Document.meta(無損側帶,最終輸出還拿得到)
        # 另補框架的 meta 契約鍵(docs/interfaces.md §2):doc_id / chunk_id
        # 讓 trace 標籤、fusion 的 group_by: doc、evaluation 直接可用。
        documents = [
            Document(
                content=hit["contentChunk"],
                score=hit.get("score"),
                meta={
                    "dockey": hit["dockey"],
                    "contentTitle": hit.get("contentTitle"),
                    "doc_id": hit["dockey"],
                    "chunk_id": f"{hit['dockey']}::chunk_{position}",
                },
            )
            for position, hit in enumerate(hits)
        ]
        logger.debug("CompanyRetriever:%r → %d 筆", query, len(documents))
        return {"documents": documents}

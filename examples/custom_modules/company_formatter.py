"""公司 formatter custom module 骨架(佔位範例,邏輯待替換)。

``formatter`` 槽位的 custom module 範本:把融合後的最終結果組成**公司
自訂格式**。這是選填的終端支線(fusion 之後,與 prompt → generation
並聯)—— 掛了之後 ``query()`` 回傳值多一個 ``output`` 鍵裝你的 payload,
canonical 的 ``documents`` 等鍵照舊(evaluation / trace 不受影響)。

槽位契約:``documents: list[Document] + query: str → payload: Any``。

三個要點:

1. **payload 的型別自由**(dict / str / 自訂類別都行)—— 這是終端槽位
   的特權:圖上沒有下游接它,唯一消費者是 query()。@component.output_types
   仍要誠實宣告你實際產出的具體型別。
2. **走 HTTP(/query 回應)時 payload 必須 JSON 可序列化**;bytes 之類
   請自行編碼(如 base64)。
3. **格式歸這裡、內容歸 meta**:公司欄位(dockey、contentTitle…)是
   custom retriever 塞進 meta 帶到這裡的 —— 兩支元件用同一套欄位命名。

掛載方式(configs/custom_demo.yaml):

    formatter:
      method: custom
      params:
        file: examples/custom_modules/company_formatter.py
        class: CompanyResponseBuilder
        # init_params:
        #   include_content: true
"""

from __future__ import annotations

import logging
from typing import Any

from haystack import component
from haystack.dataclasses import Document

# logger 命名在 "rag.*" 底下:file: 與 class_path: 兩種載入方式都吃得到
# 框架的 log 檔設定(見 README「自訂方法」的「log 要看得到」)。
logger = logging.getLogger("rag.custom.company_formatter")


@component
class CompanyResponseBuilder:
    """把最終結果組成公司回應信封(替換起點)。

    Args:
        include_content: 是否包含切片內文。
    """

    def __init__(self, include_content: bool = True) -> None:
        self.include_content = include_content

    @component.output_types(payload=dict[str, Any])
    def run(self, documents: list[Document], query: str) -> dict[str, Any]:
        """組公司信封,回傳 ``{"payload": <你的格式>}``。

        TODO(替換點):把下面的信封換成公司系統實際的回應格式。
        欄位來源:公司欄位在 custom retriever 進 meta(dockey、
        contentTitle…),這裡原樣取用 —— 取不到時給明確的 fallback,
        不要讓 KeyError 毀掉整次查詢。
        """
        rows = []
        for doc in documents:
            row: dict[str, Any] = {
                "dockey": doc.meta.get("dockey") or doc.meta.get("doc_id"),
                "contentTitle": doc.meta.get("contentTitle"),
                "score": doc.score,
            }
            if self.include_content:
                row["contentChunk"] = doc.content
            rows.append(row)
        payload = {
            "question": query,
            "totalCount": len(rows),
            "returnData": rows,
        }
        logger.debug(
            "CompanyResponseBuilder:%d 筆文件 → 信封(totalCount=%d)",
            len(documents), len(rows),
        )
        return {"payload": payload}

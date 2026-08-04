"""公司重排 custom module 骨架(佔位範例,邏輯待替換)。

``reranking`` 槽位的 custom module 範本。目前用「查詢與切片的字元重疊率」
做簡單計分(CJK 友善、離線可跑);到公司後把 ``_score`` 換成真正的
rerank 邏輯(或呼叫公司 rerank API)即可。

槽位契約:``query: str + documents: list[Document] → documents``。
上游是誰(內建 hybrid 或 custom 檢索)都一樣 —— 元件只讀
``doc.content``,需要公司欄位(dockey / contentTitle)時從 ``doc.meta``
拿,這正是「槽位間只流 canonical 型別」帶來的可組合性。

掛載方式(configs/custom_demo.yaml):

    reranking:
      method: custom
      params:
        file: examples/custom_modules/company_reranker.py
        class: KeywordOverlapReranker
        init_params: {top_k: 3}
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from haystack import component
from haystack.dataclasses import Document

# logger 命名在 "rag.*" 底下:file: 與 class_path: 兩種載入方式都吃得到
# 框架的 log 檔設定(見 README「自訂方法」的「log 要看得到」)。
logger = logging.getLogger("rag.custom.company_reranker")


@component
class KeywordOverlapReranker:
    """以查詢/切片字元重疊率重排,取前 ``top_k`` 筆。

    Args:
        top_k: 重排後保留的切片數。
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def _score(self, query: str, doc: Document) -> float:
        """計算查詢與切片的相關性分數。

        TODO(替換點):換成公司的 rerank 邏輯,例如::

            response = httpx.post(
                self.endpoint,
                json={"question": query, "documents": [d.content for d in docs]},
            )

        (批次呼叫的話,把迴圈搬進 run() 一次送出。)
        目前的佔位實作:字元集合重疊率(對中文可用,無需斷詞)。
        """
        query_chars = set(query) - set(" ,。?!?!")
        if not query_chars:
            return 0.0
        content_chars = set(doc.content or "")
        return len(query_chars & content_chars) / len(query_chars)

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document]) -> dict[str, Any]:
        reranked = []
        for doc in documents:
            # 原檢索分數先存進 meta 再覆寫 score:rerank 覆寫分數是這類
            # adapter 最常見的資訊遺失點,留著才能事後稽核。
            # 用 dataclasses.replace 而非就地賦值:Document 實例可能被
            # pipeline 其他部分共用,就地修改會產生副作用(Haystack 慣例)。
            meta = dict(doc.meta)
            if doc.score is not None and "retrieval_score" not in meta:
                meta["retrieval_score"] = doc.score
            reranked.append(
                replace(doc, score=self._score(query, doc), meta=meta)
            )
        reranked.sort(key=lambda d: d.score or 0.0, reverse=True)
        top = reranked[: self.top_k]
        logger.debug(
            "KeywordOverlapReranker:%d → %d 筆", len(documents), len(top)
        )
        return {"documents": top}

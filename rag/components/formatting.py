"""Formatter 槽位的內建方法:通用 JSON 形狀的最終輸出。

formatter 是選填的終端支線(fusion 之後,與 prompt → generation 並聯):
把融合後的文件組成**對外**格式,結果進 ``query()`` 回傳值的 ``output``
鍵。canonical 的 ``documents`` 等鍵照舊 —— formatter 是加一個鍵,
不是換掉輸出。

內建的 :class:`SimpleJsonFormatter` 產出通用 dict:離線 demo 與測試不必
寫 .py 檔,也給 custom formatter 當對照範本。公司信封請用
``method: custom``(見 examples/custom_modules/company_formatter.py)。
"""

from __future__ import annotations

from typing import Any

from haystack import Document, component


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

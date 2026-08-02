"""增量 ingest:過濾掉索引中已存在且內容未變的切片。

放在 stamper 之後、embedder 之前 —— 重點是**省下 embedding**:
chunk_id 是確定性的(``doc_id::chunk_seq``),拿全部切片的 id 去 store
批次查既有內容,內容相同的切片直接跳過,只有新增或變更的切片才往下
走 embedding → 寫入。

比對只看 ``content``(embedding 只由 content 決定;meta 變了但內容
沒變的切片不會重寫,屬可接受的取捨)。來源檔案**刪除**後的舊切片
不在此元件的守備範圍(它只看這次進來的切片),與全量 ingest 的
upsert 語意一致:要乾淨索引請刪索引重建。
"""

from __future__ import annotations

import logging
from typing import Any

from haystack import Document, component

logger = logging.getLogger(__name__)


@component
class IncrementalChangeFilter:
    """按 chunk_id 比對 store 既有內容,只放行新增或變更的切片。"""

    def __init__(self, store: Any) -> None:
        self.store = store

    @component.output_types(documents=list[Document], skipped=int)
    def run(self, documents: list[Document]) -> dict[str, Any]:
        if not documents:
            return {"documents": [], "skipped": 0}
        existing = {
            doc.id: doc.content or ""
            for doc in self.store.filter_documents(
                filters={
                    "field": "id",
                    "operator": "in",
                    "value": [doc.id for doc in documents],
                }
            )
        }
        changed = [
            doc
            for doc in documents
            if existing.get(doc.id) != (doc.content or "")
        ]
        skipped = len(documents) - len(changed)
        if skipped:
            logger.info(
                "增量 ingest:跳過 %d/%d 筆未變更的切片(不重算 embedding)",
                skipped, len(documents),
            )
        return {"documents": changed, "skipped": skipped}

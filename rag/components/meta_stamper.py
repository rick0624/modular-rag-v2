"""切片身分蓋章:doc_id / seq / page / chunk_id。

Haystack 的 ``Document.id`` 預設是內容雜湊;本框架改為顯式指定
``chunk_id = "{doc_id}::chunk_{seq}"``(v1 慣例),讓同一輸入永遠得到
同一 id —— document store 的 OVERWRITE 寫入策略因此具備 upsert 語意
(重複 ingest 不會累積重複切片),ES 的 ``_id`` 也隨之穩定。

放在 splitter 之後、embedder 之前;此後不得再改動切片內容。
"""

from __future__ import annotations

from typing import Any

from haystack import Document, component

from rag.errors import ComponentError


@component
class ChunkMetaStamper:
    """為每個切片蓋上 doc_id / seq / page / chunk_id,並過濾空白切片。"""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """依文件內出現順序編 seq,重建帶穩定 id 的 Document。

        規則(v1 契約):

        - ``seq`` 是**全文件**連續序號(不分頁重排)。
        - 空白切片不產生輸出(頁碼由 splitter 先數好,不受影響)。
        - meta 繼承上游全部欄位,再蓋上 doc_id / seq / page / chunk_id。

        Raises:
            ComponentError: 文件缺少 ``meta['doc_id']``
                (import 槽位的 FileLister 應在最上游提供)。
        """
        seq_counters: dict[str, int] = {}
        stamped: list[Document] = []
        for doc in documents:
            content = (doc.content or "").strip("\f \t\r\n")
            if not content:
                continue
            doc_id = doc.meta.get("doc_id")
            if not doc_id:
                raise ComponentError(
                    "切片缺少 meta['doc_id'],無法產生穩定的 chunk_id。"
                    "請確認 import 槽位使用會提供 doc_id 的方法(如 local_file),"
                    f"目前的 meta 欄位:{sorted(doc.meta.keys())}"
                )
            seq = seq_counters.get(doc_id, 0)
            seq_counters[doc_id] = seq + 1
            chunk_id = f"{doc_id}::chunk_{seq}"
            page = doc.meta.get("page_number")
            meta = {
                **doc.meta,
                "doc_id": doc_id,
                "seq": seq,
                "page": page if isinstance(page, int) else None,
                "chunk_id": chunk_id,
            }
            stamped.append(Document(id=chunk_id, content=content, meta=meta))
        return {"documents": stamped}

"""公司 chunking custom module 骨架(佔位範例,邏輯待替換)。

``chunking`` 槽位的 custom module 範本:把整份 Document 切成小塊。
內建方法(fixed_size / structure_based / page_based)不夠用 —— 例如
要按公司文件的節次標記切、或條文式文件一條一塊 —— 時就寫這一支。

槽位契約:``documents: list[Document] → documents: list[Document]``。

兩條紀律:

1. **meta 逐塊複製**:每個切片繼承來源 Document 的整包 meta ——
   ``doc_id`` 必須保留(下游 ChunkMetaStamper 依它產生穩定的
   ``chunk_id = "{doc_id}::chunk_{seq}"``,缺了直接報錯)。
2. **切片順序 = 文件內順序**:seq 依到達順序編號,順序亂了,重複
   ingest 的 upsert 語意(同一輸入 → 同一 chunk_id)就破了。

需要分頁輸入(頁界)時在參數宣告 ``requires_pages: true``,建構期會
檢查 parsing 鏈是否產生頁界。

掛載方式(configs/custom_ingestion_demo.yaml):

    chunking:
      method: custom
      params:
        file: examples/custom_modules/company_chunker.py
        class: CompanySentenceChunker
        init_params:
          max_chars: 120
        # requires_pages: false     # 需要頁界時設 true
"""

from __future__ import annotations

import logging
from typing import Any

from haystack import component
from haystack.dataclasses import Document

# logger 命名在 "rag.*" 底下:file: 與 class_path: 兩種載入方式都吃得到
# 框架的 log 檔設定(見 README「自訂方法」的「log 要看得到」)。
logger = logging.getLogger("rag.custom.company_chunker")


@component
class CompanySentenceChunker:
    """按句號切句、貪婪打包到字元上限(公司切塊邏輯的替換起點)。

    Args:
        max_chars: 每個切片的字元數上限。
        separator: 句子分隔符(預設中文句號)。
    """

    def __init__(self, max_chars: int = 200, separator: str = "。") -> None:
        self.max_chars = max_chars
        self.separator = separator

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """逐份文件切塊,回傳 ``{"documents": [...]}``。

        TODO(替換點):把 ``_split`` 換成公司的切塊邏輯(按節次標記、
        條文編號、表格邊界…)。
        """
        chunks: list[Document] = []
        for doc in documents:
            for piece in self._split(doc.content or ""):
                # meta 逐塊複製:doc_id 在裡面,stamper 靠它編 chunk_id
                chunks.append(Document(content=piece, meta=dict(doc.meta)))
        logger.debug(
            "CompanySentenceChunker:%d 份文件 → %d 個切片", len(documents), len(chunks)
        )
        return {"documents": chunks}

    def _split(self, text: str) -> list[str]:
        """句子貪婪打包:相鄰句子併到不超過 max_chars 為止。"""
        sentences = [
            part.strip() + self.separator
            for part in text.split(self.separator)
            if part.strip()
        ]
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > self.max_chars:
                pieces.append(current)
                current = sentence
            else:
                current += sentence
        if current:
            pieces.append(current)
        return pieces

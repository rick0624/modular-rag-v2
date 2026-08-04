"""公司 parsing custom module 骨架(佔位範例,邏輯待替換)。

``parsing`` 槽位的 custom module 範本。parsing 是方法鏈,custom 可放
兩種位置,契約不同,以參數 ``kind`` 宣告:

- **鏈首 converter**(``kind: converter``):``sources + meta → documents``
  —— 解析公司自有格式(JSON 信封、專有標記…)。本檔的
  :class:`CompanyPayloadParser`。
- **鏈中文件處理器**(``kind: doc_processor``,預設):``documents →
  documents`` —— 清理、去樣板、富化 meta。本檔的
  :class:`CompanyBoilerplateCleaner`。

一條鏈只能有一個 ``custom``(method_params 以方法名稱分區);兩種都
需要時把邏輯合併成一個元件,或一邊改走內建方法。

掛載方式:

    # 鏈首(configs/custom_ingestion_demo.yaml)
    parsing:
      method: custom
      params:
        file: examples/custom_modules/company_parser.py
        class: CompanyPayloadParser
        kind: converter
        # input_content_types: [text]   # 供與 import 的相容性檢查;省略 = 全收
        # produces_pages: false         # 產生頁界時設 true(page_based 的前提)

    # 鏈中(configs/custom_demo.yaml)
    parsing:
      method: [plain_text, custom]
      method_params:
        custom:
          file: examples/custom_modules/company_parser.py
          class: CompanyBoilerplateCleaner
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from haystack import component
from haystack.dataclasses import ByteStream, Document

# logger 命名在 "rag.*" 底下:file: 與 class_path: 兩種載入方式都吃得到
# 框架的 log 檔設定(見 README「自訂方法」的「log 要看得到」)。
logger = logging.getLogger("rag.custom.company_parser")


def _read_text(source: Any, encoding: str) -> str:
    """把一筆 source(路徑或 ByteStream)讀成文字。"""
    if isinstance(source, ByteStream):
        return source.data.decode(encoding)
    return Path(source).read_text(encoding=encoding)


@component
class CompanyPayloadParser:
    """鏈首 converter:把公司信封格式解成 Document。

    Args:
        encoding: 內容編碼。
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """逐筆解析 sources,回傳 ``{"documents": [...]}``。

        TODO(替換點):把 ``_read_text`` 之後的邏輯換成公司格式的解析
        (JSON 信封取欄位、去除專有標記…)。紀律:``meta[i]`` 要整包
        繼承進 ``documents[i].meta``(``doc_id`` 在裡面,丟了下游
        stamper 會報錯);單筆解析失敗記警告後跳過,不要讓一份壞文件
        毀掉整批 ingest。
        """
        documents: list[Document] = []
        for source, entry in zip(sources, meta):
            try:
                text = _read_text(source, self.encoding)
            except Exception as exc:
                logger.warning(
                    "CompanyPayloadParser:無法解析 %s(%s: %s),跳過",
                    entry.get("doc_id") or source, type(exc).__name__, exc,
                )
                continue
            documents.append(Document(content=text, meta=dict(entry)))
        logger.debug(
            "CompanyPayloadParser:%d 筆來源 → %d 份文件", len(sources), len(documents)
        )
        return {"documents": documents}


@component
class CompanyBoilerplateCleaner:
    """鏈中文件處理器:去除公司樣板行(頁尾、機密聲明…)。

    Args:
        patterns: 命中即整行移除的子字串清單;None 用內建示範樣板。
    """

    def __init__(self, patterns: list[str] | None = None) -> None:
        # TODO(替換點):換成公司文件實際的樣板特徵。
        self.patterns = patterns if patterns is not None else [
            "本文件為公司內部機密",
            "All Rights Reserved",
        ]

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """逐份文件濾掉樣板行,回傳 ``{"documents": [...]}``。

        紀律:回傳**新的** Document(不就地改動輸入),meta 原樣繼承。
        """
        cleaned: list[Document] = []
        removed = 0
        for doc in documents:
            lines = (doc.content or "").splitlines()
            kept = [
                line for line in lines
                if not any(pattern in line for pattern in self.patterns)
            ]
            removed += len(lines) - len(kept)
            cleaned.append(
                Document(content="\n".join(kept), meta=dict(doc.meta))
            )
        if removed:
            logger.debug("CompanyBoilerplateCleaner:移除 %d 行樣板", removed)
        return {"documents": cleaned}

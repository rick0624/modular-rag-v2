"""PDF converter:pypdf 文字層 + 選擇性 OCR。

pypdf 直接抽文字層有兩個先天限制:

1. **掃描檔(圖片型 PDF)**:文字層是空的,什麼都抽不到,檔案會
   靜默消失(切片全空白被丟棄)。
2. **多欄 / 表格版面**:pypdf 按 PDF 內部串流順序抽字,視覺上相鄰的
   內容(如榜單的「系所標題」與其下方的考生編號)在抽出的文字流中
   可能相隔數百字元,還可能重複 —— chunking 再怎麼調都救不回來。

OCR(rapidocr,需 ``pip install -e ".[ocr]"``)按**視覺位置**輸出文字,
同時解掉這兩個問題。``mode`` 三檔:

- ``off``:純 pypdf(原 PyPDFToDocument 行為)。
- ``auto``(預設):逐頁判斷 —— 文字層有內容用文字層,空的(掃描頁)
  才 OCR。未安裝 OCR 時記警告並跳過該頁(至少不再靜默)。
- ``force``:全部頁面走 OCR(版面順序亂的 PDF 用;成本約每頁數秒)。

輸出與 PyPDFToDocument 相容:每檔一個 Document、頁與頁之間以 ``\\f``
分隔(下游 splitter 據此標頁碼)、meta 沿用 ByteStream / meta 清單。
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Literal

from haystack import Document, component
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)

OcrMode = Literal["off", "auto", "force"]


def _ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        import pypdfium2  # noqa: F401

        return True
    except ImportError:
        return False


@component
class PdfToDocument:
    """PDF → Document(pypdf 文字層 + 依 ``mode`` 選擇性 OCR)。"""

    def __init__(self, mode: OcrMode = "auto", ocr_scale: float = 2.0) -> None:
        """
        Args:
            mode: OCR 策略(off / auto / force,見模組 docstring)。
            ocr_scale: OCR 前的頁面渲染倍率(越大越清楚也越慢)。
        """
        self.mode = mode
        self.ocr_scale = ocr_scale
        self._ocr_engine: Any = None  # 首次用到才載入(模型初始化要幾秒)

    def _ocr_page_texts(self, data: bytes, page_indices: list[int]) -> dict[int, str]:
        """對指定頁面做 OCR,回傳 {頁索引: 文字}。"""
        import numpy as np
        import pypdfium2 as pdfium
        from rapidocr_onnxruntime import RapidOCR

        if self._ocr_engine is None:
            self._ocr_engine = RapidOCR()
        texts: dict[int, str] = {}
        pdf = pdfium.PdfDocument(data)
        try:
            for index in page_indices:
                image = pdf[index].render(scale=self.ocr_scale).to_pil()
                result, _ = self._ocr_engine(np.array(image))
                # rapidocr 依偵測框位置(由上而下)輸出 → 保留視覺順序
                texts[index] = "\n".join(text for _, text, _ in (result or []))
        finally:
            pdf.close()
        return texts

    def _convert(self, bytestream: ByteStream, label: str) -> str | None:
        """單一 PDF → 全文(頁以 \\f 分隔);完全讀不到時回 None。"""
        from pypdf import PdfReader

        data = bytestream.data
        try:
            reader = PdfReader(io.BytesIO(data))
            page_texts = [
                (page.extract_text() or "").strip() for page in reader.pages
            ]
        except Exception as exc:
            logger.warning("無法讀取 PDF %s(%s: %s),跳過", label, type(exc).__name__, exc)
            return None

        need_ocr = [
            index
            for index, text in enumerate(page_texts)
            if self.mode == "force" or (self.mode == "auto" and not text)
        ]
        if need_ocr:
            if _ocr_available():
                logger.info(
                    "PDF %s:%d/%d 頁走 OCR(mode=%s)",
                    label, len(need_ocr), len(page_texts), self.mode,
                )
                for index, text in self._ocr_page_texts(data, need_ocr).items():
                    page_texts[index] = text
            elif self.mode == "force":  # 退回文字層,至少不比 off 差
                logger.warning(
                    "PDF %s 設定 ocr: force 但未安裝 OCR,退回 pypdf 文字層。"
                    "安裝:pip install -e \".[ocr]\"", label,
                )
            else:
                logger.warning(
                    "PDF %s 有 %d 頁沒有文字層(掃描頁),但未安裝 OCR;"
                    "這些頁面將是空的。安裝:pip install -e \".[ocr]\"",
                    label, len(need_ocr),
                )
        return "\f".join(page_texts)

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        meta_list = normalize_metadata(meta, sources_count=len(sources))
        documents: list[Document] = []
        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as exc:
                logger.warning(
                    "無法讀取來源 %s(%s: %s),跳過", source, type(exc).__name__, exc
                )
                continue
            merged_meta = {**bytestream.meta, **metadata}
            label = str(merged_meta.get("doc_id") or source)
            content = self._convert(bytestream, label)
            if content is None:
                continue
            documents.append(Document(content=content, meta=merged_meta))
        return {"documents": documents}

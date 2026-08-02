"""共用測試 fixture:離線語料、最小 PDF、config 樣板、假 HTTP client。

測試鐵則(v1 文化):任何測試都不碰網路、不下載模型。
需要 HTTP 的元件一律以可注入的 FakeClient 驗證。
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

BASE_CONFIG: dict[str, Any] = {
    "ingestion": {
        # extensions 釘住純文字:讓既有測試持續走 plain_text 路徑
        # (local_file 預設收 txt/md/pdf → mixed,那需要 auto parsing)
        "import": {
            "method": "local_file",
            "params": {"input_dir": "./data/raw", "extensions": [".txt", ".md"]},
        },
        "parsing": {"method": "plain_text"},
        "chunking": {"method": "fixed_size"},
        "embedding": {"method": "mock", "params": {"dim": 16}},
        "indexing": {"method": "in_memory"},
    },
    "inference": {
        "query_transformation": {"method": "passthrough"},
        "retrieval": {"method": "bm25"},
        "reranking": {"method": "llm"},
        "generation": {"method": "mock"},
    },
}


def make_config(
    ingestion: dict[str, Any] | None = None,
    inference: dict[str, Any] | None = None,
    **top_level: Any,
) -> dict[str, Any]:
    """回傳 BASE_CONFIG 的深拷貝,依槽位覆寫。"""
    data = copy.deepcopy(BASE_CONFIG)
    for slot, value in (ingestion or {}).items():
        data["ingestion"][slot] = value
    for slot, value in (inference or {}).items():
        data["inference"][slot] = value
    data.update(top_level)
    return data


@pytest.fixture
def corpus_dir(tmp_path):
    """建立一小批繁中文字語料(含子資料夾,驗證 doc_id 相對路徑)。"""
    raw = tmp_path / "raw"
    (raw / "sub").mkdir(parents=True)
    (raw / "faiss.txt").write_text(
        "FAISS 是 Facebook 開源的向量檢索函式庫,支援多種索引結構與 GPU 加速。\n\n"
        "它常用於大規模相似度搜尋,與 sentence-transformers 搭配建立語意檢索。",
        encoding="utf-8",
    )
    (raw / "sub" / "es.txt").write_text(
        "Elasticsearch 是分散式搜尋引擎,原生支援 BM25 關鍵字檢索與 kNN 向量檢索。\n\n"
        "混合檢索(hybrid)把兩路結果以 RRF 融合,兼顧語意與精確匹配。",
        encoding="utf-8",
    )
    return raw


def minimal_two_page_pdf(page1: str, page2: str) -> bytes:
    """手刻最小可解析的 2 頁 PDF(ASCII 文字;測 page 流程用,不需外部工具)。"""
    objs: list[bytes] = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>")
    page_tpl = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents %d 0 R /Resources << /Font << /F1 7 0 R >> >> >>"
    )
    objs.append(page_tpl % 5)
    objs.append(page_tpl % 6)
    for text in (page1, page2):
        stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
        objs.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (index, body)
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        xref_pos,
    )
    return bytes(out)


@pytest.fixture
def pdf_dir(tmp_path):
    """建立含一份 2 頁 PDF 的資料夾。"""
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "manual.pdf").write_bytes(
        minimal_two_page_pdf("Alpha vector search page", "Beta keyword ranking page")
    )
    return pdfs


@pytest.fixture
def mixed_corpus_dir(corpus_dir):
    """txt(含子資料夾)+ md + pdf 混放的語料夾(測 auto 分流)。"""
    (corpus_dir / "notes.md").write_text(
        "# RAG 筆記\n\n檢索增強生成先檢索相關內容,再交給 LLM 作答。",
        encoding="utf-8",
    )
    (corpus_dir / "manual.pdf").write_bytes(
        minimal_two_page_pdf("Alpha vector search page", "Beta keyword ranking page")
    )
    return corpus_dir


class FakeResponse:
    """最小 httpx.Response 替身。"""

    def __init__(
        self, status_code: int = 200, payload: Any = None, text: str = ""
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("回應不是 JSON")
        return self._payload


class FakeClient:
    """最小 httpx.Client 替身:記錄每次呼叫,依序回傳預備的回應。

    元素是 Exception 時改為拋出(模擬 timeout / 連線錯誤)。
    """

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

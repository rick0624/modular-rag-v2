"""PdfToDocument(pypdf + 選擇性 OCR)與空來源回報的測試。

不依賴 rapidocr 是否安裝:掃描頁(空白頁)即使 OCR 也讀不出字,
觀察到的行為(空來源回報)兩種環境一致;需要固定行為的測試以
monkeypatch 關閉 OCR 可用性。
"""

from __future__ import annotations

import logging

from conftest import make_config, minimal_two_page_pdf

import rag.components.pdf_ocr as pdf_ocr_module
from rag.builder import build_pipelines
from rag.components.pdf_ocr import PdfToDocument
from rag.config import parse_config


def _scanned_pdf_dir(tmp_path):
    """「掃描檔」替身:有頁面結構但文字層是空的。"""
    pdfs = tmp_path / "scanned"
    pdfs.mkdir()
    (pdfs / "scan.pdf").write_bytes(minimal_two_page_pdf("", ""))
    return pdfs


def _config(input_dir, **parsing_params):
    return parse_config(
        make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(input_dir), "extensions": [".pdf"]},
                },
                "parsing": {"method": "pdf", "params": parsing_params},
            }
        )
    )


def test_text_layer_pdf_unchanged_by_default(pdf_dir):
    """有文字層的 PDF:auto 模式走文字層,行為與原 PyPDFToDocument 一致。"""
    pipelines = build_pipelines(_config(pdf_dir))
    result = pipelines.run_ingestion()
    assert result["writer"]["documents_written"] > 0
    assert result["empty_sources"] == []
    docs = pipelines.store.filter_documents()
    assert {d.meta["page"] for d in docs} <= {1, 2}


def test_scanned_pdf_reported_as_empty_source_without_ocr(
    tmp_path, monkeypatch, caplog
):
    """掃描檔 + 無 OCR:檔案不再靜默消失,警告與 empty_sources 都要出現。"""
    monkeypatch.setattr(pdf_ocr_module, "_ocr_available", lambda: False)
    pipelines = build_pipelines(_config(_scanned_pdf_dir(tmp_path)))
    with caplog.at_level(logging.WARNING):
        result = pipelines.run_ingestion()
    assert result["empty_sources"] == ["scan.pdf"]
    assert result.get("writer", {}).get("documents_written", 0) == 0
    joined = " ".join(record.message for record in caplog.records)
    assert ".[ocr]" in joined  # 警告要指路:裝 OCR extra
    assert "沒有產出任何切片" in joined


def test_mode_off_never_touches_ocr(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        pdf_ocr_module, "_ocr_available", lambda: calls.append(1) or True
    )
    pipelines = build_pipelines(_config(_scanned_pdf_dir(tmp_path), ocr="off"))
    result = pipelines.run_ingestion()
    assert calls == []  # off:連可用性都不必查
    assert result["empty_sources"] == ["scan.pdf"]


def test_force_without_ocr_falls_back_to_text_layer(pdf_dir, monkeypatch, caplog):
    """force 但沒裝 OCR:退回文字層(至少不比 off 差),並警告。"""
    monkeypatch.setattr(pdf_ocr_module, "_ocr_available", lambda: False)
    pipelines = build_pipelines(_config(pdf_dir, ocr="force"))
    with caplog.at_level(logging.WARNING):
        result = pipelines.run_ingestion()
    assert result["writer"]["documents_written"] > 0  # 文字層內容還在
    assert any("退回 pypdf 文字層" in r.message for r in caplog.records)


def test_unreadable_pdf_skipped_with_warning(tmp_path, caplog):
    pdfs = tmp_path / "bad"
    pdfs.mkdir()
    (pdfs / "broken.pdf").write_bytes(b"not a pdf at all")
    pipelines = build_pipelines(_config(pdfs))
    with caplog.at_level(logging.WARNING):
        result = pipelines.run_ingestion()
    assert result["empty_sources"] == ["broken.pdf"]


def test_yaml_boolean_ocr_values_accepted(pdf_dir):
    """YAML 把裸寫的 off/on 解析成布林;直接接受,不強迫使用者加引號。"""
    import yaml

    from rag.methods_ingestion import _PdfParams
    from rag.errors import ConfigError

    assert yaml.safe_load("ocr: off")["ocr"] is False  # 前提:確實變成布林
    assert _PdfParams(ocr=False).ocr == "off"
    assert _PdfParams(ocr=True).ocr == "force"
    assert _PdfParams(ocr="auto").ocr == "auto"

    # 仍要擋掉真正的錯字
    try:
        build_pipelines(_config(pdf_dir, ocr="nope"))
    except ConfigError as exc:
        assert "'off', 'auto' or 'force'" in str(exc)
    else:
        raise AssertionError("無效的 ocr 值應報錯")


def test_component_direct_bytestream_meta_flow(pdf_dir):
    """ByteStream 路徑(auto 分流)的 meta 合併:doc_id 要跟著進 Document。"""
    from haystack.dataclasses import ByteStream

    data = (pdf_dir / "manual.pdf").read_bytes()
    stream = ByteStream(data=data, meta={"doc_id": "manual.pdf"})
    out = PdfToDocument(mode="off").run(sources=[stream])
    assert len(out["documents"]) == 1
    doc = out["documents"][0]
    assert doc.meta["doc_id"] == "manual.pdf"
    assert "\f" in doc.content  # 頁界保留給下游 splitter

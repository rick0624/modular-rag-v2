"""auto parsing(混合 KB)測試:content-type 推導、分流、meta 契約。"""

from __future__ import annotations

import pytest
from conftest import make_config

from rag.builder import build_ingestion_pipeline, build_pipelines
from rag.methods_ingestion import _local_file_output_type
from rag.config import parse_config
from rag.errors import ConfigError, IncompatiblePipelineError


def _auto_config(input_dir, *, parsing="auto", chunking=None, extensions=None):
    import_params = {"input_dir": str(input_dir)}
    if extensions is not None:
        import_params["extensions"] = extensions
    return parse_config(
        make_config(
            ingestion={
                "import": {"method": "local_file", "params": import_params},
                "parsing": {"method": parsing},
                **({"chunking": chunking} if chunking else {}),
            }
        )
    )


class TestContentTypeDerivation:
    def test_homogeneous_text(self):
        assert _local_file_output_type({"extensions": [".txt"]}) == "text"
        assert _local_file_output_type({"extensions": [".md"]}) == "text"
        assert _local_file_output_type({"extensions": [".txt", ".md"]}) == "text"

    def test_homogeneous_pdf(self):
        assert _local_file_output_type({"extensions": [".pdf"]}) == "pdf"

    def test_default_is_mixed(self):
        assert _local_file_output_type({}) == "mixed"
        assert _local_file_output_type({"extensions": None}) == "mixed"

    def test_case_insensitive(self):
        assert _local_file_output_type({"extensions": [".TXT", ".Pdf"]}) == "mixed"

    def test_unknown_extension_rejected(self):
        with pytest.raises(ConfigError, match=r"不支援副檔名.*\.docx"):
            _local_file_output_type({"extensions": [".docx"]})


class TestCompatibility:
    def test_default_local_file_with_plain_text_rejected_suggests_auto(self, corpus_dir):
        config = _auto_config(corpus_dir, parsing="plain_text")
        with pytest.raises(IncompatiblePipelineError) as excinfo:
            build_ingestion_pipeline(config)
        assert "content_type {'mixed'}" in str(excinfo.value)
        assert "'auto'" in str(excinfo.value)

    def test_auto_with_page_based_chunking_validates(self, pdf_dir):
        config = _auto_config(pdf_dir, chunking={"method": "page_based"})
        build_ingestion_pipeline(config)  # 不應丟例外(auto 產生分頁)


class TestAutoParsing:
    def test_mixed_corpus_end_to_end(self, mixed_corpus_dir):
        # 切長要小於單頁內容,頁界(\f)才會生效、pdf 切片各自帶頁碼
        _, store = _run_ingestion(
            _auto_config(
                mixed_corpus_dir,
                chunking={
                    "method": "fixed_size",
                    "params": {"split_length": 40, "split_overlap": 0},
                },
            )
        )
        docs = store.filter_documents()
        doc_ids = {d.meta["doc_id"] for d in docs}
        # txt(含子資料夾)、md、pdf 全部進同一個 KB
        assert doc_ids == {"faiss.txt", "sub/es.txt", "notes.md", "manual.pdf"}
        for doc in docs:
            meta = doc.meta
            # meta 契約:doc_id 經 ByteStream 流進各 converter 後仍在
            assert doc.id == meta["chunk_id"] == f"{meta['doc_id']}::chunk_{meta['seq']}"
            assert meta["importer"] == "local_file"
        # pdf 帶真頁碼;文字檔為 page=1(既有語意)
        pdf_pages = {d.meta["page"] for d in docs if d.meta["doc_id"] == "manual.pdf"}
        assert pdf_pages == {1, 2}
        text_pages = {d.meta["page"] for d in docs if d.meta["doc_id"] != "manual.pdf"}
        assert text_pages == {1}

    def test_markdown_only_corpus_routes_correctly(self, tmp_path):
        """Windows 的 mimetypes 不認得 .md;此測試守住 additional_mimetypes。"""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "only.md").write_text("# 標題\n\n內文段落。", encoding="utf-8")
        _, store = _run_ingestion(_auto_config(raw))
        assert {d.meta["doc_id"] for d in store.filter_documents()} == {"only.md"}

    def test_text_only_corpus_through_auto(self, corpus_dir):
        """沒有 pdf 時 router 的 pdf socket 不觸發,joiner 照常合流。"""
        _, store = _run_ingestion(_auto_config(corpus_dir))
        assert {d.meta["doc_id"] for d in store.filter_documents()} == {
            "faiss.txt",
            "sub/es.txt",
        }

    def test_auto_clean_chain(self, mixed_corpus_dir):
        config = parse_config(
            make_config(
                ingestion={
                    "import": {
                        "method": "local_file",
                        "params": {"input_dir": str(mixed_corpus_dir)},
                    },
                    "parsing": {
                        "method": ["auto", "clean"],
                        "method_params": {"clean": {"remove_extra_whitespaces": True}},
                    },
                }
            )
        )
        _, store = _run_ingestion(config)
        assert store.count_documents() > 0

    def test_reingest_upserts_without_duplicates(self, mixed_corpus_dir):
        config = _auto_config(mixed_corpus_dir)
        pipeline, store = build_ingestion_pipeline(config)
        pipeline.run({})
        count = store.count_documents()
        pipeline2, _ = build_ingestion_pipeline(config, store=store)
        pipeline2.run({})
        assert store.count_documents() == count

    def test_seq_deterministic_across_runs(self, mixed_corpus_dir):
        """分支合流不影響 chunk_id:兩次獨立 ingest 產出完全相同的切片身分。"""
        config = _auto_config(mixed_corpus_dir)

        def snapshot():
            pipeline, store = build_ingestion_pipeline(config)
            pipeline.run({})
            return {
                (d.meta["doc_id"], d.meta["seq"]): d.content
                for d in store.filter_documents()
            }

        assert snapshot() == snapshot()

    def test_trace_tolerates_router_step(self, mixed_corpus_dir):
        from rag.trace import format_ingestion_trace

        pipelines = build_pipelines(_auto_config(mixed_corpus_dir))
        result = pipelines.run_ingestion()
        by_name = {entry["component"]: entry for entry in result["trace"]}
        assert by_name["parser_router"]["count"] is None  # ByteStream 分流步驟
        assert by_name["parser_join"]["count"] == 4  # 三個 converter 合流(4 份文件)
        lines = format_ingestion_trace(result["trace"])
        assert any("parser_router" in line and "—" in line for line in lines)


def _run_ingestion(config):
    pipeline, store = build_ingestion_pipeline(config)
    pipeline.run({})
    return pipeline, store

"""ingestion 端到端測試(離線):身分蓋章、meta 繼承、upsert、page 流程。"""

from __future__ import annotations

import pytest
from conftest import make_config

from rag.builder import build_ingestion_pipeline
from rag.config import parse_config


def _build_and_run(config_dict):
    config = parse_config(config_dict)
    pipeline, store = build_ingestion_pipeline(config)
    pipeline.run({})
    return config, pipeline, store


class TestTextIngestion:
    def test_chunk_identity_and_meta(self, corpus_dir):
        _, _, store = _build_and_run(
            make_config(
                ingestion={
                    "import": {
                        "method": "local_file",
                        "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                    },
                    "chunking": {
                        "method": "fixed_size",
                        "params": {"split_length": 40, "split_overlap": 0},
                    },
                }
            )
        )
        docs = store.filter_documents()
        assert docs, "應有切片寫入 store"
        for doc in docs:
            meta = doc.meta
            # chunk_id 慣例與 Document.id 一致(穩定 upsert 的關鍵)
            assert doc.id == meta["chunk_id"] == f"{meta['doc_id']}::chunk_{meta['seq']}"
            # meta 繼承:FileLister 提供的來源資訊要流到每個切片
            assert meta["importer"] == "local_file"
            assert meta["source"].endswith(".txt")
            assert doc.embedding is not None and len(doc.embedding) == 16
            assert doc.content.strip(), "空白切片不應產生"
        # 子資料夾檔案的 doc_id 是相對 POSIX 路徑
        doc_ids = {d.meta["doc_id"] for d in docs}
        assert doc_ids == {"faiss.txt", "sub/es.txt"}

    def test_seq_is_contiguous_per_document(self, corpus_dir):
        _, _, store = _build_and_run(
            make_config(
                ingestion={
                    "import": {
                        "method": "local_file",
                        "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                    },
                    "chunking": {
                        "method": "fixed_size",
                        "params": {"split_length": 30, "split_overlap": 0},
                    },
                }
            )
        )
        by_doc: dict[str, list[int]] = {}
        for doc in store.filter_documents():
            by_doc.setdefault(doc.meta["doc_id"], []).append(doc.meta["seq"])
        for doc_id, seqs in by_doc.items():
            assert sorted(seqs) == list(range(len(seqs))), (
                f"{doc_id} 的 seq 應為 0..N-1 連續:{sorted(seqs)}"
            )
        assert any(len(seqs) > 1 for seqs in by_doc.values()), (
            "中文語料在 char 模式下應被切成多個切片"
        )

    def test_reingest_upserts_without_duplicates(self, corpus_dir):
        config_dict = make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                },
            }
        )
        config, _, store = _build_and_run(config_dict)
        first_count = store.count_documents()
        pipeline2, _ = build_ingestion_pipeline(config, store=store)
        pipeline2.run({})
        assert store.count_documents() == first_count

    def test_structure_based_splits_on_paragraphs(self, corpus_dir):
        _, _, store = _build_and_run(
            make_config(
                ingestion={
                    "import": {
                        "method": "local_file",
                        "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                    },
                    "chunking": {
                        "method": "structure_based",
                        "params": {"split_length": 60},
                    },
                }
            )
        )
        faiss_chunks = [
            d for d in store.filter_documents() if d.meta["doc_id"] == "faiss.txt"
        ]
        assert len(faiss_chunks) >= 2, "兩個段落應至少切成兩個切片"


class TestPdfIngestion:
    def _pdf_config(self, pdf_dir, chunking):
        return make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(pdf_dir), "extensions": [".pdf"]},
                },
                "parsing": {"method": "pdf"},
                "chunking": chunking,
            }
        )

    def test_page_based_chunks_carry_page_numbers(self, pdf_dir):
        _, _, store = _build_and_run(
            self._pdf_config(pdf_dir, {"method": "page_based"})
        )
        docs = sorted(store.filter_documents(), key=lambda d: d.meta["seq"])
        assert [d.meta["page"] for d in docs] == [1, 2]
        assert [d.id for d in docs] == [
            "manual.pdf::chunk_0",
            "manual.pdf::chunk_1",
        ]
        assert "Alpha" in docs[0].content and "Beta" in docs[1].content
        # 頁界符號應被清掉,不殘留在切片內容
        assert "\f" not in docs[0].content

    def test_char_chunking_still_tracks_pages(self, pdf_dir):
        _, _, store = _build_and_run(
            self._pdf_config(
                pdf_dir,
                {"method": "fixed_size", "params": {"split_length": 30, "split_overlap": 0}},
            )
        )
        pages = {d.meta["page"] for d in store.filter_documents()}
        assert pages == {1, 2}, "字元切分下 splitter 仍應標記來源頁碼"

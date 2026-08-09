"""增量 ingest 測試:未變切片跳過 embedding、新增/變更切片照常寫入。"""

from __future__ import annotations

import pytest
from conftest import make_config

from rag.builder import INDEXING_FACTORIES, SlotFactory, build_pipelines
from rag.config import parse_config
from rag.errors import IncompatiblePipelineError


def _incremental_config(corpus_dir):
    return parse_config(
        make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                },
                "indexing": {"method": "in_memory", "params": {"incremental": True}},
            }
        )
    )


def _counts(result):
    return {entry["component"]: entry["count"] for entry in result["trace"]}


def test_second_ingest_skips_everything(corpus_dir):
    pipelines = build_pipelines(_incremental_config(corpus_dir))
    first_result = pipelines.run_ingestion()
    first = _counts(first_result)
    assert first["source_filter"] == first["importer"]  # 首跑:全部檔案放行
    assert first["change_filter"] == first["stamper"]
    assert first["embedder"] == first["stamper"]

    second = pipelines.run_ingestion()
    counts = _counts(second)
    # 檔案層:兩個檔案都沒變,連 parse 都跳過
    assert counts["source_filter"] == 0
    assert set(second["source_filter"]["skipped_files"]) == {"faiss.txt", "sub/es.txt"}
    assert counts["embedder"] == 0
    assert second["writer"]["documents_written"] == 0
    assert second["empty_sources"] == []  # 被跳過 ≠ 空來源,不可誤報


def test_new_file_only_its_chunks_pass(corpus_dir):
    pipelines = build_pipelines(_incremental_config(corpus_dir))
    pipelines.run_ingestion()
    before = pipelines.store.count_documents()

    (corpus_dir / "new.txt").write_text("Qdrant 是向量資料庫。", encoding="utf-8")
    result = pipelines.run_ingestion()

    passed = result["change_filter"]["documents"]
    assert {doc.meta["doc_id"] for doc in passed} == {"new.txt"}  # 只有新檔案
    assert pipelines.store.count_documents() > before
    # 舊檔案的切片仍在,且沒有被重寫成重複
    doc_ids = {d.meta["doc_id"] for d in pipelines.store.filter_documents()}
    assert doc_ids == {"faiss.txt", "sub/es.txt", "new.txt"}


def test_modified_file_chunks_pass_again(corpus_dir):
    pipelines = build_pipelines(_incremental_config(corpus_dir))
    pipelines.run_ingestion()

    (corpus_dir / "faiss.txt").write_text(
        "FAISS 改版了:現在也支援 GPU 上的 IVF-PQ 索引。", encoding="utf-8"
    )
    result = pipelines.run_ingestion()

    passed = result["change_filter"]["documents"]
    assert passed, "變更檔案的切片應重新放行"
    assert {doc.meta["doc_id"] for doc in passed} == {"faiss.txt"}
    # store 中的內容確實被更新
    contents = " ".join(
        d.content for d in pipelines.store.filter_documents()
        if d.meta["doc_id"] == "faiss.txt"
    )
    assert "IVF-PQ" in contents


def test_incremental_requires_capability(corpus_dir, monkeypatch):
    """索引未宣告 incremental_update 時,建構期就報不相容。"""
    original = INDEXING_FACTORIES["in_memory"]
    monkeypatch.setitem(
        INDEXING_FACTORIES,
        "in_memory",
        SlotFactory(
            build=original.build,
            capabilities=original.capabilities - {"incremental_update"},
        ),
    )
    with pytest.raises(IncompatiblePipelineError, match="incremental_update"):
        build_pipelines(_incremental_config(corpus_dir))


def test_parse_config_change_invalidates_manifest(corpus_dir):
    """chunking 等設定變更後,檔案沒變也必須全量重 parse(manifest 作廢)。"""
    pipelines = build_pipelines(_incremental_config(corpus_dir))
    pipelines.run_ingestion()

    changed = make_config(
        ingestion={
            "import": {
                "method": "local_file",
                "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
            },
            "chunking": {
                "method": "fixed_size",
                "params": {"split_length": 77, "split_overlap": 0},
            },
            "indexing": {"method": "in_memory", "params": {"incremental": True}},
        }
    )
    from rag.builder import build_ingestion_pipeline
    from rag.config import parse_config as _parse

    # 同一個 store(模擬服務內換設定後 /ingest)
    pipeline2, _ = build_ingestion_pipeline(_parse(changed), store=pipelines.store)
    output = pipeline2.run({}, include_outputs_from={"source_filter"})
    assert output["source_filter"]["skipped_files"] == []  # key 不符 → 不跳過


def test_incremental_off_by_default(corpus_dir):
    config = parse_config(
        make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                },
            }
        )
    )
    pipelines = build_pipelines(config)
    result = pipelines.run_ingestion()
    assert "change_filter" not in {e["component"] for e in result["trace"]}


# ---------------------------------------------------------------------------
# custom import × 檔案層增量:非路徑 sources 優雅退化
# ---------------------------------------------------------------------------

BYTESTREAM_IMPORTER = '''
from pathlib import Path
from typing import Any
from haystack import component
from haystack.dataclasses import ByteStream


@component
class StreamImporter:
    @component.output_types(
        sources=list[str | Path | ByteStream], meta=list[dict[str, Any]]
    )
    def run(self) -> dict[str, Any]:
        return {
            "sources": [ByteStream(data="請假需三日前申請。".encode("utf-8"))],
            "meta": [{"doc_id": "API-1"}],
        }
'''

PATH_IMPORTER = '''
from pathlib import Path
from typing import Any
from haystack import component
from haystack.dataclasses import ByteStream


@component
class PathImporter:
    """custom importer 也可以回傳真實路徑:檔案層增量照常運作。"""

    def __init__(self, input_dir: str) -> None:
        self.input_dir = input_dir

    @component.output_types(
        sources=list[str | Path | ByteStream], meta=list[dict[str, Any]]
    )
    def run(self) -> dict[str, Any]:
        files = sorted(Path(self.input_dir).rglob("*.txt"))
        return {
            "sources": [str(f) for f in files],
            "meta": [
                {"doc_id": f.relative_to(self.input_dir).as_posix()} for f in files
            ],
        }
'''

CONVERTER = '''
from pathlib import Path
from typing import Any
from haystack import component
from haystack.dataclasses import ByteStream, Document


@component
class AnySourceParser:
    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: list[dict[str, Any]],
    ) -> dict[str, Any]:
        documents = []
        for source, entry in zip(sources, meta):
            text = (
                source.data.decode("utf-8")
                if isinstance(source, ByteStream)
                else Path(source).read_text(encoding="utf-8")
            )
            documents.append(Document(content=text, meta=dict(entry)))
        return {"documents": documents}
'''


def _custom_import_incremental_config(tmp_path, importer_source, cls, init_params=None):
    importer = tmp_path / "importer.py"
    importer.write_text(importer_source, encoding="utf-8")
    converter = tmp_path / "converter.py"
    converter.write_text(CONVERTER, encoding="utf-8")
    params = {"file": str(importer), "class": cls}
    if init_params:
        params["init_params"] = init_params
    return parse_config(
        make_config(
            ingestion={
                "import": {"method": "custom", "params": params},
                "parsing": {
                    "method": "custom",
                    "params": {
                        "file": str(converter),
                        "class": "AnySourceParser",
                        "kind": "converter",
                    },
                },
                "indexing": {"method": "in_memory", "params": {"incremental": True}},
            }
        )
    )


def test_non_path_sources_always_reparse(tmp_path, caplog):
    """ByteStream 來源無從比對雜湊:每次全量重 parse,且不記入 manifest。"""
    config = _custom_import_incremental_config(
        tmp_path, BYTESTREAM_IMPORTER, "StreamImporter"
    )
    pipelines = build_pipelines(config)
    pipelines.run_ingestion()

    import logging

    caplog.clear()  # 只驗證第二輪的警告數
    with caplog.at_level(logging.WARNING, logger="rag.components.ingestion_steps"):
        second = pipelines.run_ingestion()
    assert second["source_filter"]["skipped_files"] == []  # 沒有檔案層跳過
    assert second["source_filter"]["manifest"]["files"] == {}  # 不記入 manifest
    warnings = [r for r in caplog.records if "不是本地檔案路徑" in r.message]
    assert len(warnings) == 1  # 彙總一次,不逐筆洗版
    # 切片層增量仍有效:內容沒變 → embedding 被跳過
    assert second["change_filter"]["skipped"] == 1


def test_custom_importer_with_real_paths_still_increments(tmp_path, corpus_dir):
    """custom import × incremental 不被禁止:回傳真路徑時檔案層照常跳過。"""
    config = _custom_import_incremental_config(
        tmp_path, PATH_IMPORTER, "PathImporter",
        init_params={"input_dir": str(corpus_dir)},
    )
    pipelines = build_pipelines(config)
    pipelines.run_ingestion()
    second = pipelines.run_ingestion()
    assert set(second["source_filter"]["skipped_files"]) == {"faiss.txt", "sub/es.txt"}


def test_custom_module_content_change_invalidates_manifest(tmp_path, corpus_dir):
    """custom parsing 的 .py 內容變了 → manifest key 變 → 全量重 parse。"""
    config = _custom_import_incremental_config(
        tmp_path, PATH_IMPORTER, "PathImporter",
        init_params={"input_dir": str(corpus_dir)},
    )
    pipelines = build_pipelines(config)
    pipelines.run_ingestion()

    # 改 converter 的 .py(語意等價的內容變更即可觸發)
    converter = tmp_path / "converter.py"
    converter.write_text(CONVERTER + "\n# parsing 邏輯改版\n", encoding="utf-8")

    from rag.builder import build_ingestion_pipeline

    pipeline2, _ = build_ingestion_pipeline(config, store=pipelines.store)
    output = pipeline2.run({}, include_outputs_from={"source_filter"})
    assert output["source_filter"]["skipped_files"] == []  # 檔案沒變也不跳過

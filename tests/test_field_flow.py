"""欄位流測試:chunking 生成欄位 → source_field embedding → fields 白名單
→ 增量比對。涵蓋單元(三個元件)與端到端(custom chunker + 增量)。"""

from __future__ import annotations

import pytest
from conftest import make_config
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from rag.builder import build_ingestion_pipeline, build_pipelines
from rag.components.ingestion_steps import (
    FieldSourceEmbedder,
    IncrementalChangeFilter,
    MetaFieldMapper,
)
from rag.components.mock_embedders import MockDocumentEmbedder, mock_vector
from rag.config import parse_config
from rag.errors import ComponentError, ConfigError

# 整檔一個切片,另生成 summary(取第一行)與 department 欄位。
SUMMARY_CHUNKER = '''
from typing import Any
from haystack import Document, component


@component
class SummaryChunker:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        out = []
        for doc in documents:
            summary = (doc.content or "").splitlines()[0].strip()
            meta = {**doc.meta, "summary": summary, "department": "kb"}
            out.append(Document(content=doc.content, meta=meta))
        return {"documents": out}
'''


# ---------------------------------------------------------------------------
# FieldSourceEmbedder
# ---------------------------------------------------------------------------


class TestFieldSourceEmbedder:
    @staticmethod
    def _doc(chunk_id: str, content: str, **meta):
        return Document(
            id=chunk_id, content=content, meta={"chunk_id": chunk_id, **meta}
        )

    def test_embeds_meta_field_and_keeps_document(self):
        embedder = FieldSourceEmbedder(MockDocumentEmbedder(dim=16), "summary")
        doc = self._doc("a::chunk_0", "內容甲", summary="摘要一")
        out = embedder.run([doc])["documents"][0]
        assert out.embedding == mock_vector("摘要一", 16)  # 向量來自欄位
        assert out.content == "內容甲"  # content 不變
        assert out.meta["summary"] == "摘要一"

    @pytest.mark.parametrize("bad_value", [None, "", "   ", 123])
    def test_missing_or_invalid_field_fails_with_chunk_id(self, bad_value):
        embedder = FieldSourceEmbedder(MockDocumentEmbedder(dim=16), "summary")
        meta = {} if bad_value is None else {"summary": bad_value}
        doc = self._doc("a::chunk_0", "內容", **meta)
        with pytest.raises(ComponentError, match="source_field 'summary'") as excinfo:
            embedder.run([doc])
        assert "a::chunk_0" in str(excinfo.value)

    def test_warm_up_delegates_to_inner(self):
        class Inner:
            warmed = False

            def warm_up(self):
                self.warmed = True

            def run(self, documents):
                return {"documents": documents}

        inner = Inner()
        FieldSourceEmbedder(inner, "summary").warm_up()
        assert inner.warmed

    def test_empty_documents(self):
        embedder = FieldSourceEmbedder(MockDocumentEmbedder(dim=16), "summary")
        assert embedder.run([])["documents"] == []

    def test_extra_vectors_written_to_meta(self):
        embedder = FieldSourceEmbedder(
            MockDocumentEmbedder(dim=16),
            "content",
            extra_vectors={"summary_vector": "summary"},
        )
        doc = self._doc("a::chunk_0", "內容甲", summary="摘要一")
        out = embedder.run([doc])["documents"][0]
        assert out.embedding == mock_vector("內容甲", 16)  # 主向量來自 content
        assert out.meta["summary_vector"] == mock_vector("摘要一", 16)
        assert out.content == "內容甲" and out.meta["summary"] == "摘要一"

    def test_extra_vectors_combine_with_source_field(self):
        # 主向量吃 summary、額外向量吃 content:兩者可交叉
        embedder = FieldSourceEmbedder(
            MockDocumentEmbedder(dim=16),
            "summary",
            extra_vectors={"content_vector": "content"},
        )
        doc = self._doc("a::chunk_0", "內容甲", summary="摘要一")
        out = embedder.run([doc])["documents"][0]
        assert out.embedding == mock_vector("摘要一", 16)
        assert out.meta["content_vector"] == mock_vector("內容甲", 16)

    def test_missing_extra_source_fails_with_chunk_id(self):
        embedder = FieldSourceEmbedder(
            MockDocumentEmbedder(dim=16),
            "content",
            extra_vectors={"summary_vector": "summary"},
        )
        doc = self._doc("a::chunk_0", "內容")
        with pytest.raises(
            ComponentError, match="extra_vectors 來源欄位 'summary'"
        ) as excinfo:
            embedder.run([doc])
        assert "a::chunk_0" in str(excinfo.value)


# ---------------------------------------------------------------------------
# MetaFieldMapper
# ---------------------------------------------------------------------------


class TestMetaFieldMapper:
    def test_whitelist_rename_and_framework_fields_kept(self):
        doc = Document(
            id="a::chunk_0",
            content="內容",
            embedding=[0.1, 0.2],
            meta={
                "doc_id": "a",
                "chunk_id": "a::chunk_0",
                "seq": 0,
                "page": 1,
                "summary": "摘要",
                "department": "kb",  # 未列入白名單 → 丟棄
                "source": "/tmp/a.txt",  # 同上
            },
        )
        out = MetaFieldMapper({"summary_text": "summary"}).run([doc])["documents"][0]
        assert out.meta == {
            "doc_id": "a",
            "chunk_id": "a::chunk_0",
            "seq": 0,
            "page": 1,
            "summary_text": "摘要",
        }
        assert out.content == "內容" and out.embedding == [0.1, 0.2]

    def test_missing_mapped_field_is_omitted(self):
        doc = Document(content="內容", meta={"doc_id": "b"})
        out = MetaFieldMapper({"summary_text": "summary"}).run([doc])["documents"][0]
        assert out.meta == {"doc_id": "b"}  # 不塞 None

    def test_preserve_keeps_vector_fields_through_whitelist(self):
        # extra_vectors 的向量欄位由框架生成,不列白名單也要原名保留
        doc = Document(
            content="內容",
            meta={"doc_id": "a", "summary": "摘要", "summary_vector": [0.1, 0.2]},
        )
        out = MetaFieldMapper(
            {"summary_text": "summary"}, preserve=("summary_vector",)
        ).run([doc])["documents"][0]
        assert out.meta["summary_vector"] == [0.1, 0.2]
        assert out.meta["summary_text"] == "摘要"
        assert "summary" not in out.meta


# ---------------------------------------------------------------------------
# IncrementalChangeFilter × source_field
# ---------------------------------------------------------------------------


class TestChangeFilterSourceField:
    @staticmethod
    def _store_with(content: str, **meta):
        store = InMemoryDocumentStore()
        store.write_documents(
            [Document(id="a::chunk_0", content=content, meta=meta)]
        )
        return store

    @staticmethod
    def _current(content: str, **meta):
        return Document(id="a::chunk_0", content=content, meta=meta)

    def test_field_change_releases_even_if_content_same(self):
        store = self._store_with("內容", summary="舊摘要")
        f = IncrementalChangeFilter(store, source_field="summary")
        f.stored_field = "summary"
        out = f.run([self._current("內容", summary="新摘要")])
        assert out["skipped"] == 0 and len(out["documents"]) == 1

    def test_content_change_releases_even_if_field_same(self):
        store = self._store_with("舊內容", summary="摘要")
        f = IncrementalChangeFilter(store, source_field="summary", stored_field="summary")
        out = f.run([self._current("新內容", summary="摘要")])
        assert out["skipped"] == 0 and len(out["documents"]) == 1

    def test_both_unchanged_skips(self):
        store = self._store_with("內容", summary="摘要")
        f = IncrementalChangeFilter(store, source_field="summary", stored_field="summary")
        out = f.run([self._current("內容", summary="摘要")])
        assert out["skipped"] == 1 and out["documents"] == []

    def test_stored_field_rename_lookup(self):
        # store 端的欄位已被 fields 改名為 summary_text
        store = self._store_with("內容", summary_text="摘要")
        f = IncrementalChangeFilter(
            store, source_field="summary", stored_field="summary_text"
        )
        out = f.run([self._current("內容", summary="摘要")])
        assert out["skipped"] == 1  # 改名對照後判定未變更

    def test_missing_current_value_releases(self):
        # current 缺值一律放行:讓下游 embedder 的 fail-fast 出聲
        store = self._store_with("內容")
        f = IncrementalChangeFilter(store, source_field="summary", stored_field="summary")
        out = f.run([self._current("內容")])
        assert out["skipped"] == 0 and len(out["documents"]) == 1

    def test_default_params_keep_content_only_semantics(self):
        store = self._store_with("內容", summary="舊")
        f = IncrementalChangeFilter(store)
        out = f.run([self._current("內容", summary="新")])
        assert out["skipped"] == 1  # 預設只看 content,meta 變化不觸發

    def test_extra_field_change_releases(self):
        store = self._store_with("內容", summary="舊摘要")
        f = IncrementalChangeFilter(store, extra_fields=[("summary", "summary")])
        out = f.run([self._current("內容", summary="新摘要")])
        assert out["skipped"] == 0 and len(out["documents"]) == 1

    def test_extra_field_unchanged_skips(self):
        store = self._store_with("內容", summary="摘要")
        f = IncrementalChangeFilter(store, extra_fields=[("summary", "summary")])
        out = f.run([self._current("內容", summary="摘要")])
        assert out["skipped"] == 1 and out["documents"] == []

    def test_extra_field_rename_lookup(self):
        # store 端的額外來源欄位已被 fields 改名為 summary_text
        store = self._store_with("內容", summary_text="摘要")
        f = IncrementalChangeFilter(
            store, extra_fields=[("summary", "summary_text")]
        )
        out = f.run([self._current("內容", summary="摘要")])
        assert out["skipped"] == 1

    def test_missing_extra_value_releases(self):
        store = self._store_with("內容")
        f = IncrementalChangeFilter(store, extra_fields=[("summary", "summary")])
        out = f.run([self._current("內容")])
        assert out["skipped"] == 0 and len(out["documents"]) == 1


# ---------------------------------------------------------------------------
# 建構期欄位引用檢查
# ---------------------------------------------------------------------------


def _field_config(
    tmp_path,
    corpus_dir,
    *,
    provides=("summary", "department"),
    source_field="summary",
    extra_vectors=None,
    indexing_params=None,
):
    chunker = tmp_path / "chunker.py"
    chunker.write_text(SUMMARY_CHUNKER, encoding="utf-8")
    chunking_params = {"file": str(chunker), "class": "SummaryChunker"}
    if provides is not None:
        chunking_params["provides_fields"] = list(provides)
    embedding_params = {"dim": 16, "source_field": source_field}
    if extra_vectors is not None:
        embedding_params["extra_vectors"] = extra_vectors
    return parse_config(
        make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                },
                "chunking": {"method": "custom", "params": chunking_params},
                "embedding": {"method": "mock", "params": embedding_params},
                "indexing": {
                    "method": "in_memory",
                    "params": dict(indexing_params or {}),
                },
            }
        )
    )


class TestBuildTimeFieldChecks:
    def test_undeclared_source_field_rejected(self, tmp_path, corpus_dir):
        config = _field_config(tmp_path, corpus_dir, source_field="sumary")  # 打錯字
        with pytest.raises(ConfigError, match="source_field 'sumary'") as excinfo:
            build_ingestion_pipeline(config)
        assert "'summary'" in str(excinfo.value)  # 列出可用欄位

    def test_undeclared_fields_meta_name_rejected(self, tmp_path, corpus_dir):
        config = _field_config(
            tmp_path, corpus_dir,
            indexing_params={"fields": {"summary_text": "summry"}},  # 打錯字
        )
        with pytest.raises(ConfigError, match="未宣告的 meta 欄位 'summry'"):
            build_ingestion_pipeline(config)

    def test_no_declaration_skips_build_time_check(self, tmp_path, corpus_dir):
        # 未宣告 provides_fields → 不做建構期檢查(退化為執行期報錯)
        config = _field_config(
            tmp_path, corpus_dir, provides=None, source_field="anything"
        )
        build_ingestion_pipeline(config)  # 不應拋錯

    def test_incremental_requires_embedded_field_in_whitelist(
        self, tmp_path, corpus_dir
    ):
        config = _field_config(
            tmp_path, corpus_dir,
            indexing_params={
                "incremental": True,
                "fields": {"dept": "department"},  # 白名單漏掉 summary
            },
        )
        with pytest.raises(ConfigError, match="fields 白名單必須包含該欄位"):
            build_ingestion_pipeline(config)

    def test_provides_fields_rejects_reserved_names(self, tmp_path, corpus_dir):
        config = _field_config(
            tmp_path, corpus_dir, provides=("summary", "chunk_id")
        )
        with pytest.raises(ConfigError, match="框架保留欄位"):
            build_ingestion_pipeline(config)

    def test_fields_rejects_reserved_es_name(self, tmp_path, corpus_dir):
        config = _field_config(
            tmp_path, corpus_dir,
            indexing_params={"fields": {"content": "summary"}},
        )
        with pytest.raises(ConfigError, match="框架保留名"):
            build_ingestion_pipeline(config)

    def test_fields_rejects_duplicate_meta_names(self, tmp_path, corpus_dir):
        config = _field_config(
            tmp_path, corpus_dir,
            indexing_params={"fields": {"a": "summary", "b": "summary"}},
        )
        with pytest.raises(ConfigError, match="不可重複"):
            build_ingestion_pipeline(config)


class TestExtraVectorsChecks:
    def test_reserved_vector_name_rejected(self, tmp_path, corpus_dir):
        config = _field_config(
            tmp_path, corpus_dir, extra_vectors={"embedding": "summary"}
        )
        with pytest.raises(ConfigError, match="框架保留名"):
            build_ingestion_pipeline(config)

    def test_duplicate_sources_rejected(self, tmp_path, corpus_dir):
        config = _field_config(
            tmp_path, corpus_dir,
            extra_vectors={"v1": "summary", "v2": "summary"},
        )
        with pytest.raises(ConfigError, match="來源欄位不可重複"):
            build_ingestion_pipeline(config)

    def test_vector_name_colliding_with_source_rejected(self, tmp_path, corpus_dir):
        config = _field_config(
            tmp_path, corpus_dir,
            source_field="content",
            extra_vectors={"summary": "summary"},  # 向量會覆蓋原文欄位
        )
        with pytest.raises(ConfigError, match="與來源欄位重名"):
            build_ingestion_pipeline(config)

    def test_undeclared_extra_source_rejected(self, tmp_path, corpus_dir):
        config = _field_config(
            tmp_path, corpus_dir,
            extra_vectors={"summary_vector": "sumary"},  # 打錯字
        )
        with pytest.raises(
            ConfigError, match="extra_vectors 來源欄位 'sumary'"
        ) as excinfo:
            build_ingestion_pipeline(config)
        assert "'summary'" in str(excinfo.value)  # 列出可用欄位

    def test_vector_name_colliding_with_declared_field_rejected(
        self, tmp_path, corpus_dir
    ):
        config = _field_config(
            tmp_path, corpus_dir,
            source_field="content",
            extra_vectors={"department": "summary"},  # 撞到 chunker 宣告的欄位
        )
        with pytest.raises(ConfigError, match="與.*既有欄位重名"):
            build_ingestion_pipeline(config)

    def test_fields_may_reference_vector_fields(self, tmp_path, corpus_dir):
        # 向量欄位可列入 fields 白名單(改名用),不算「未宣告」
        config = _field_config(
            tmp_path, corpus_dir,
            source_field="content",
            extra_vectors={"summary_vector": "summary"},
            indexing_params={"fields": {"sum_vec": "summary_vector"}},
        )
        build_ingestion_pipeline(config)  # 不應拋錯

    def test_incremental_requires_extra_sources_in_whitelist(
        self, tmp_path, corpus_dir
    ):
        config = _field_config(
            tmp_path, corpus_dir,
            source_field="content",
            extra_vectors={"summary_vector": "summary"},
            indexing_params={
                "incremental": True,
                "fields": {"dept": "department"},  # 白名單漏掉 summary
            },
        )
        with pytest.raises(ConfigError, match="fields 白名單必須包含該欄位"):
            build_ingestion_pipeline(config)


# ---------------------------------------------------------------------------
# 端到端:custom chunker 欄位 → source_field embedding → 白名單 → 增量
# ---------------------------------------------------------------------------


def test_field_flow_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # 兩份文件第一行(= summary)相同、內文不同:驗證向量只由 summary 決定
    (raw / "a.txt").write_text("共同摘要\n\n甲文件的內文。", encoding="utf-8")
    (raw / "b.txt").write_text("共同摘要\n\n乙文件的內文,完全不同。", encoding="utf-8")

    config = _field_config(
        tmp_path, raw,
        indexing_params={
            "incremental": True,
            "fields": {"summary_text": "summary", "dept": "department"},
        },
    )
    pipelines = build_pipelines(config, stage="ingestion")

    # 首跑:全部寫入,embedding 來自 summary、meta 依白名單改名
    result1 = pipelines.run_ingestion()
    assert result1["writer"]["documents_written"] == 2
    docs = pipelines.store.filter_documents()
    for doc in docs:
        assert doc.embedding == mock_vector("共同摘要", 16)
        assert doc.meta["summary_text"] == "共同摘要"
        assert doc.meta["dept"] == "kb"
        assert "department" not in doc.meta and "source" not in doc.meta
        assert {"doc_id", "chunk_id", "seq"} <= set(doc.meta)
    # 兩份內文不同的文件,summary 相同 → 向量相同
    assert docs[0].embedding == docs[1].embedding

    # 重跑無變更:檔案層增量全跳過,不重寫
    result2 = pipelines.run_ingestion()
    assert result2["writer"]["documents_written"] == 0

    # 改第一行(summary 變)→ 該文件重 embed,向量跟著新 summary
    (raw / "a.txt").write_text("新的摘要\n\n甲文件的內文。", encoding="utf-8")
    result3 = pipelines.run_ingestion()
    assert result3["writer"]["documents_written"] == 1
    changed = next(
        d for d in pipelines.store.filter_documents() if d.meta["doc_id"] == "a.txt"
    )
    assert changed.embedding == mock_vector("新的摘要", 16)
    assert changed.meta["summary_text"] == "新的摘要"


def test_extra_vectors_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("摘要行\n\n甲文件的內文。", encoding="utf-8")

    config = _field_config(
        tmp_path, raw,
        source_field="content",
        extra_vectors={"summary_vector": "summary"},
        indexing_params={
            "incremental": True,
            "fields": {"summary_text": "summary"},  # 額外來源欄位需在白名單
        },
    )
    pipelines = build_pipelines(config, stage="ingestion")

    # 首跑:主向量來自 content,額外向量來自 summary 且不受白名單影響
    result1 = pipelines.run_ingestion()
    assert result1["writer"]["documents_written"] == 1
    doc = pipelines.store.filter_documents()[0]
    assert doc.embedding == mock_vector(doc.content, 16)
    assert doc.meta["summary_vector"] == mock_vector("摘要行", 16)
    assert doc.meta["summary_text"] == "摘要行"
    assert "summary" not in doc.meta  # 原欄位已依白名單改名

    # 重跑無變更:全跳過,不重寫
    result2 = pipelines.run_ingestion()
    assert result2["writer"]["documents_written"] == 0


def test_extra_vectors_renamed_via_fields(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("摘要行\n\n內文。", encoding="utf-8")

    config = _field_config(
        tmp_path, raw,
        source_field="content",
        extra_vectors={"summary_vector": "summary"},
        indexing_params={
            "fields": {
                "sum_vec": "summary_vector",   # 向量欄位改名
                "summary_text": "summary",
            },
        },
    )
    pipelines = build_pipelines(config, stage="ingestion")
    pipelines.run_ingestion()
    doc = pipelines.store.filter_documents()[0]
    assert doc.meta["sum_vec"] == mock_vector("摘要行", 16)
    assert "summary_vector" not in doc.meta  # 已改名,不重複保留原名

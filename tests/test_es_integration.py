"""Elasticsearch 整合測試(選配,預設排除)。

    docker compose up -d
    pip install -e ".[dev,es]"
    ES_URL=http://localhost:9200 python -m pytest -m es

叢集有開 security 時,另外帶 ES_API_KEY 或 ES_USERNAME/ES_PASSWORD
(必要時 ES_CA_CERTS);沒帶會收到 401 missing authentication credentials。
"""

from __future__ import annotations

import os
import uuid

import pytest
from conftest import make_config

from rag.builder import build_ingestion_pipeline, build_pipelines
from rag.config import parse_config

ES_URL = os.environ.get("ES_URL")

pytestmark = [
    pytest.mark.es,
    pytest.mark.skipif(not ES_URL, reason="需要 ES_URL 指向執行中的 Elasticsearch"),
]


def _auth_params() -> dict:
    """指向有開 security 的叢集時,從環境變數帶認證(否則會 401)。"""
    params = {}
    if os.environ.get("ES_API_KEY"):
        params["api_key"] = os.environ["ES_API_KEY"]
    elif os.environ.get("ES_USERNAME"):
        params["username"] = os.environ["ES_USERNAME"]
        params["password"] = os.environ.get("ES_PASSWORD", "")
    if os.environ.get("ES_CA_CERTS"):
        params["ca_certs"] = os.environ["ES_CA_CERTS"]
    return params


@pytest.fixture
def es_config_dict(corpus_dir):
    index = f"modular-rag-test-{uuid.uuid4().hex[:8]}"
    return make_config(
        ingestion={
            "import": {"method": "local_file", "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]}},
            "indexing": {
                "method": "elasticsearch",
                "params": {"hosts": ES_URL, "index": index, **_auth_params()},
            },
        },
        inference={
            "retrieval": {"method": "hybrid", "params": {"top_k": 4}},
            "reranking": {"method": "none"},
        },
    )


def _cleanup(store, index_name: str) -> None:
    try:
        store.client.indices.delete(index=index_name)
    except Exception:
        pass  # 清理失敗不影響測試結果(索引名帶 uuid,不會互相污染)


def test_es_ingest_query_and_upsert(es_config_dict):
    config = parse_config(es_config_dict)
    index_name = config.ingestion.indexing.params_for("elasticsearch")["index"]
    pipelines = build_pipelines(config)
    try:
        pipelines.run_ingestion()
        first_count = pipelines.store.count_documents()
        assert first_count > 0

        # hybrid(server 端 BM25 + kNN,RRF 融合)
        result = pipelines.query("混合檢索怎麼融合兩路結果?")
        assert result["documents"], "ES hybrid 檢索應有結果"
        top = result["documents"][0]
        assert top.meta["chunk_id"] == top.id
        assert top.meta["doc_id"] in {"faiss.txt", "sub/es.txt"}

        # 重複 ingest:Document.id = chunk_id → ES _id 穩定,upsert 不重複
        pipeline2, _ = build_ingestion_pipeline(config, store=pipelines.store)
        pipeline2.run({})
        pipelines.store.client.indices.refresh(index=index_name)
        assert pipelines.store.count_documents() == first_count
    finally:
        _cleanup(pipelines.store, index_name)


def test_es_fingerprint_roundtrip(es_config_dict):
    """指紋寫進 ES index mapping 的 _meta,跨 store 實例讀得回來。"""
    from rag.kb_meta import ingestion_fingerprint, read_fingerprint, write_fingerprint

    config = parse_config(es_config_dict)
    index_name = config.ingestion.indexing.params_for("elasticsearch")["index"]
    pipelines = build_pipelines(config)
    try:
        pipelines.run_ingestion()  # 索引建立後才能寫 mapping
        fingerprint = ingestion_fingerprint(es_config_dict)
        write_fingerprint("elasticsearch", pipelines.store, fingerprint, index_name)
        assert (
            read_fingerprint("elasticsearch", pipelines.store, index_name)
            == fingerprint
        )
        # 全新 store 實例(模擬服務重啟)也讀得到:指紋跟著索引走
        fresh = build_pipelines(config)
        assert (
            read_fingerprint("elasticsearch", fresh.store, index_name) == fingerprint
        )
    finally:
        _cleanup(pipelines.store, index_name)


def test_es_fingerprint_missing_index_returns_none(es_config_dict):
    from rag.kb_meta import read_fingerprint

    config = parse_config(es_config_dict)
    pipelines = build_pipelines(config)
    index_name = config.ingestion.indexing.params_for("elasticsearch")["index"]
    try:
        assert (
            read_fingerprint("elasticsearch", pipelines.store, "no-such-index-xyz")
            is None
        )
    finally:
        _cleanup(pipelines.store, index_name)


def test_es_incremental_ingest_skips_unchanged(es_config_dict, corpus_dir):
    es_config_dict["ingestion"]["indexing"]["params"]["incremental"] = True
    config = parse_config(es_config_dict)
    index_name = config.ingestion.indexing.params_for("elasticsearch")["index"]
    pipelines = build_pipelines(config)
    try:
        first = pipelines.run_ingestion()
        first_written = first["writer"]["documents_written"]
        assert first_written > 0
        pipelines.store.client.indices.refresh(index=index_name)

        # 第二次:內容沒變,change_filter 全部跳過,不重算 embedding
        second = pipelines.run_ingestion()
        assert second["writer"]["documents_written"] == 0
        assert second["change_filter"]["skipped"] == first_written

        # 新增檔案:只有新檔的切片通過
        (corpus_dir / "extra.txt").write_text("增量測試新檔。", encoding="utf-8")
        third = pipelines.run_ingestion()
        passed = third["change_filter"]["documents"]
        assert {d.meta["doc_id"] for d in passed} == {"extra.txt"}
    finally:
        _cleanup(pipelines.store, index_name)


def test_es_bm25_only(es_config_dict):
    es_config_dict["inference"]["retrieval"] = {
        "method": "bm25",
        "params": {"top_k": 3},
    }
    config = parse_config(es_config_dict)
    index_name = config.ingestion.indexing.params_for("elasticsearch")["index"]
    pipelines = build_pipelines(config)
    try:
        pipelines.run_ingestion()
        result = pipelines.query("FAISS 支援哪些索引結構?")
        assert result["documents"]
        assert result["documents"][0].meta["doc_id"] == "faiss.txt"
    finally:
        _cleanup(pipelines.store, index_name)

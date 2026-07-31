"""Elasticsearch 整合測試(選配,預設排除)。

    docker compose up -d
    pip install -e ".[dev,es]"
    ES_URL=http://localhost:9200 python -m pytest -m es
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


@pytest.fixture
def es_config_dict(corpus_dir):
    index = f"modular-rag-test-{uuid.uuid4().hex[:8]}"
    return make_config(
        ingestion={
            "import": {"method": "local_file", "params": {"input_dir": str(corpus_dir)}},
            "indexing": {
                "method": "elasticsearch",
                "params": {"hosts": ES_URL, "index": index},
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

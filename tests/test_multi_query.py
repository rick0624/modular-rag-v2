"""MultiQueryRetrievalStage 與 builder 的多子查詢接線測試。"""

from __future__ import annotations

from conftest import make_config

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from rag.builder import build_inference_pipeline, build_pipelines
from rag.components.mock_embedders import MockTextEmbedder, mock_vector
from rag.components.multi_query import MultiQueryRetrievalStage
from rag.config import parse_config


def _store_with(texts: dict[str, str], dim: int = 16) -> InMemoryDocumentStore:
    store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    store.write_documents(
        [
            Document(
                id=f"{doc_id}::chunk_0",
                content=text,
                embedding=mock_vector(text, dim),
                meta={"doc_id": doc_id, "chunk_id": f"{doc_id}::chunk_0", "seq": 0},
            )
            for doc_id, text in texts.items()
        ]
    )
    return store


def test_stage_runs_inner_pipeline_per_query():
    store = _store_with({"faiss": "FAISS 向量檢索", "es": "Elasticsearch 關鍵字檢索"})
    inner = Pipeline()
    inner.add_component("query_embedder", MockTextEmbedder(dim=16))
    inner.add_component(
        "retriever", InMemoryEmbeddingRetriever(document_store=store, top_k=1)
    )
    inner.connect("query_embedder.embedding", "retriever.query_embedding")
    stage = MultiQueryRetrievalStage(
        inner=inner,
        query_targets=[("query_embedder", "text")],
        output_component="retriever",
    )
    out = stage.run(queries=["FAISS 向量", "Elasticsearch 關鍵字"])
    assert len(out["results"]) == 2
    assert out["results"][0][0].meta["doc_id"] == "faiss"
    assert out["results"][1][0].meta["doc_id"] == "es"


def test_builder_wires_rerank_chain_into_inner_pipeline():
    config = parse_config(
        make_config(
            inference={
                "retrieval": {"method": "hybrid"},
                "reranking": {
                    "method": ["llm", "llm"],
                    "method_params": {
                        "llm": {
                            "top_k": 3,
                            "generator": {
                                "method": "mock",
                                "params": {"replies": ['{"documents": [{"index": 1}]}']},
                            },
                        }
                    },
                },
            }
        )
    )
    pipeline, meta = build_inference_pipeline(config)
    stage = pipeline.get_component("multi_query")
    inner_components = set(stage.inner.to_dict()["components"].keys())
    assert {"query_embedder", "bm25_retriever", "embedding_retriever", "joiner"} <= inner_components
    assert {"ranker", "ranker_2"} <= inner_components, "雙段 rerank 應是兩個節點"
    assert stage.output_component == "ranker_2"
    # 查詢文字要餵給向量端、BM25 端與兩個 ranker
    assert set(stage.query_targets) == {
        ("query_embedder", "text"),
        ("bm25_retriever", "query"),
        ("ranker", "query"),
        ("ranker_2", "query"),
    }
    assert meta["query_entry"] == ("multi_query", "queries")


def test_query_embedder_derived_from_ingestion_embedding():
    """同向量空間紀律:查詢端 embedder 一律派生自 ingestion.embedding。"""
    config = parse_config(
        make_config(
            ingestion={"embedding": {"method": "mock", "params": {"dim": 8}}},
            inference={"retrieval": {"method": "embedding"}},
        )
    )
    pipelines = build_pipelines(config)
    stage = pipelines.inference.get_component("multi_query")
    query_embedder = stage.inner.get_component("query_embedder")
    assert isinstance(query_embedder, MockTextEmbedder)
    assert query_embedder.dim == 8, "查詢端維度必須跟 ingestion 設定一致"

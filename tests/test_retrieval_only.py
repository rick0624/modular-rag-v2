"""檢索-only 端到端測試:condense 流程全離線跑完。

一次驗證:jargon_mapping 替換、llm_rewrite 改寫(腳本)、hybrid 檢索
候選放大(boost_k_factor)、LLM 重排(腳本)、llm_fact_check 過濾
(腳本)、fusion 去重排序取 top 3、generate_answer: false 跳過生成。
"""

from __future__ import annotations

from conftest import make_config

from rag.builder import build_pipelines
from rag.config import parse_config
from rag.evaluation import run_evaluation


def _condense_config(corpus_dir, **top_level):
    data = make_config(
            **top_level,
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                },
            },
            inference={
                "query_transformation": {
                    "method": ["jargon_mapping", "llm_rewrite"],
                    "method_params": {
                        "jargon_mapping": {
                            "mapping": {"名次融合": "RRF"},
                        },
                        "llm_rewrite": {
                            "generator": {
                                "method": "mock",
                                "params": {
                                    "replies": [
                                        '{"rewritten_query": "混合檢索 RRF 融合",'
                                        ' "reason": "改寫成檢索友善形式"}'
                                    ]
                                },
                            }
                        },
                    },
                },
                "retrieval": {
                    "method": "hybrid",
                    "params": {"top_k": 2, "boost_k_factor": 3},
                },
                "reranking": {
                    "method": ["llm", "llm_fact_check"],
                    "method_params": {
                        "llm": {
                            "top_k": 4,
                            "generator": {
                                "method": "mock",
                                "params": {
                                    # LLMRanker 協定:1 起編號,未列出者丟棄
                                    "replies": ['{"documents": [{"index": 1}, {"index": 2}]}']
                                },
                            },
                        },
                        "llm_fact_check": {
                            "generator": {
                                "method": "mock",
                                "params": {"replies": ['{"relevant_indices": [1, 2]}']},
                            }
                        },
                    },
                },
                "generate_answer": False,
                "fusion": {"group_by": "none", "strategy": "rrf", "top_k": 3},
            },
    )
    del data["inference"]["generation"]  # 檢索-only:LLM 方法各自帶 generator
    return parse_config(data)


def test_retrieval_only_full_run(corpus_dir):
    config = _condense_config(corpus_dir)
    assert config.inference.generation is None  # 檢索-only 可完全省略 generation

    pipelines = build_pipelines(config)
    ingestion_result = pipelines.run_ingestion()
    assert ingestion_result["writer"]["documents_written"] > 0

    result = pipelines.query("混合檢索的名次融合怎麼運作?")

    # 生成相關欄位一律為 None,key 形狀不變
    assert result["answer"] is None
    assert result["prompt"] is None
    assert result["reply_meta"] is None

    # 檢索結果:top 3、去重(group_key = chunk_id)、分數降冪
    documents = result["documents"]
    assert 0 < len(documents) <= 3
    assert len({d.meta["group_key"] for d in documents}) == len(documents)
    for doc in documents:
        assert doc.meta["group_key"] == doc.meta["chunk_id"]
    assert [d.score for d in documents] == sorted(
        (d.score for d in documents), reverse=True
    ), "融合結果必須降冪"

    # 單一查詢(rewrite 是 1→1,不拆解)
    assert len(result["subquery_results"]) == 1


def test_retrieval_only_evaluation_works(corpus_dir):
    config = _condense_config(
        corpus_dir,
        evaluation={
            "method": "basic_retrieval_metrics",
            "params": {
                "cases": [
                    {"query": "向量檢索函式庫", "relevant_doc_ids": ["faiss.txt"]}
                ]
            },
        },
    )
    pipelines = build_pipelines(config)
    pipelines.run_ingestion()

    evaluation = run_evaluation(pipelines, config)
    assert evaluation["num_cases"] == 1
    assert 0.0 <= evaluation["metrics"]["hit_rate"] <= 1.0

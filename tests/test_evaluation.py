"""評估測試:JSONL 載入、指標計算、config 檔解析煙測。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_config

from haystack import Document

from rag.config import load_config, parse_config
from rag.errors import ComponentError, UnknownMethodError
from rag.evaluation import (
    EvalCase,
    RetrievalMetricsEvaluator,
    build_evaluator,
    load_cases,
    run_evaluation,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestLoadCases:
    def test_loads_and_skips_blank_lines(self, tmp_path):
        path = tmp_path / "qa.jsonl"
        path.write_text(
            '{"query": "q1", "relevant_doc_ids": ["a"]}\n\n'
            '{"query": "q2", "relevant_doc_ids": ["b"], "reference_answer": "ans"}\n',
            encoding="utf-8",
        )
        cases = load_cases(path)
        assert [case.query for case in cases] == ["q1", "q2"]
        assert cases[1].reference_answer == "ans"

    def test_bad_line_reports_line_number(self, tmp_path):
        path = tmp_path / "qa.jsonl"
        path.write_text(
            '{"query": "q1", "relevant_doc_ids": ["a"]}\nnot json\n', encoding="utf-8"
        )
        with pytest.raises(ComponentError, match="第 2 行"):
            load_cases(path)

    def test_empty_dataset_rejected(self, tmp_path):
        path = tmp_path / "qa.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ComponentError, match="沒有任何測試案例"):
            load_cases(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ComponentError, match="找不到評估資料集"):
            load_cases(tmp_path / "nope.jsonl")


def _result(*doc_ids: str) -> dict:
    return {
        "documents": [
            Document(id=f"{doc_id}::chunk_{i}", content="x", meta={"doc_id": doc_id})
            for i, doc_id in enumerate(doc_ids)
        ]
    }


class TestMetrics:
    def test_hit_rate_and_mrr(self):
        evaluator = RetrievalMetricsEvaluator(cases=[EvalCase(query="q", relevant_doc_ids=["x"])])
        cases = [
            EvalCase(query="q1", relevant_doc_ids=["a"]),
            EvalCase(query="q2", relevant_doc_ids=["b"]),
            EvalCase(query="q3", relevant_doc_ids=["z"]),
        ]
        results = [
            _result("a", "c"),       # rank 1 → rr = 1
            _result("c", "d", "b"),  # rank 3 → rr = 1/3
            _result("c", "d"),       # 未命中 → rr = 0
        ]
        out = evaluator.evaluate(cases, results)
        assert out["num_cases"] == 3
        assert out["metrics"]["hit_rate"] == pytest.approx(2 / 3)
        assert out["metrics"]["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
        assert out["per_query"][1]["retrieved_doc_ids"] == ["c", "d", "b"]

    def test_duplicate_doc_chunks_count_once_in_rank_order(self):
        evaluator = RetrievalMetricsEvaluator(cases=[EvalCase(query="q", relevant_doc_ids=["x"])])
        cases = [EvalCase(query="q", relevant_doc_ids=["b"])]
        # 文件 a 有兩個切片排前面;去重後 b 應是第 2 名 → rr = 1/2
        results = [
            {
                "documents": [
                    Document(id="a::chunk_0", content="x", meta={"doc_id": "a"}),
                    Document(id="a::chunk_1", content="x", meta={"doc_id": "a"}),
                    Document(id="b::chunk_0", content="x", meta={"doc_id": "b"}),
                ]
            }
        ]
        out = evaluator.evaluate(cases, results)
        assert out["per_query"][0]["retrieved_doc_ids"] == ["a", "b"]
        assert out["per_query"][0]["reciprocal_rank"] == pytest.approx(0.5)


class TestBuildEvaluator:
    def test_unknown_method_lists_alternatives(self):
        config = parse_config(
            make_config(
                evaluation={"method": "ragas", "params": {}}
            )
        )
        with pytest.raises(UnknownMethodError, match="'basic_retrieval_metrics'"):
            build_evaluator(config)

    def test_inline_cases(self):
        config = parse_config(
            make_config(
                evaluation={
                    "method": "basic_retrieval_metrics",
                    "params": {
                        "cases": [{"query": "q", "relevant_doc_ids": ["a"]}]
                    },
                }
            )
        )
        evaluator = build_evaluator(config)
        assert [case.query for case in evaluator.load_cases()] == ["q"]


class TestEndToEndEvaluation:
    def test_run_evaluation_over_smoke_pipelines(self, corpus_dir, tmp_path):
        from rag.builder import build_pipelines

        qa = tmp_path / "qa.jsonl"
        qa.write_text(
            json.dumps(
                {"query": "FAISS 支援哪些索引?", "relevant_doc_ids": ["faiss.txt"]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        config = parse_config(
            make_config(
                ingestion={
                    "import": {
                        "method": "local_file",
                        "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                    }
                },
                evaluation={
                    "method": "basic_retrieval_metrics",
                    "params": {"dataset_path": str(qa)},
                },
            )
        )
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()
        out = run_evaluation(pipelines, config)
        assert out["num_cases"] == 1
        assert out["metrics"]["hit_rate"] == 1.0, "BM25 對關鍵字查詢應命中 faiss.txt"


class TestConfigFilesParse:
    """configs/ 內的檔案要能通過 schema 驗證(公司檔以假環境變數解析)。"""

    def test_default_yaml_parses(self):
        config = load_config(CONFIGS_DIR / "default.yaml", dotenv_path=None)
        assert config.ingestion.embedding.method == "mock"
        # 型錄性質:每個槽位都要並存展示所有可用方法
        from rag import builder

        for cfg, table in [
            (config.ingestion.import_, builder.IMPORT_FACTORIES),
            (config.ingestion.parsing, builder.PARSING_FACTORIES),
            (config.ingestion.chunking, builder.CHUNKING_FACTORIES),
            (config.ingestion.embedding, builder.EMBEDDING_FACTORIES),
            (config.ingestion.indexing, builder.INDEXING_FACTORIES),
            (config.inference.query_transformation, builder.TRANSFORM_FACTORIES),
            (config.inference.retrieval, builder.RETRIEVAL_FACTORIES),
            (config.inference.reranking, builder.RERANKING_FACTORIES),
            (config.inference.generation, builder.GENERATION_FACTORIES),
        ]:
            missing = set(table) - set(cfg.method_params)
            assert not missing, f"default.yaml 型錄缺少方法區塊:{sorted(missing)}"

    def test_all_commented_block_is_treated_as_empty(self):
        """參數全被註解掉的區塊在 YAML 中是 null,不該驗證失敗(型錄常見)。"""
        config = parse_config(
            make_config(
                ingestion={
                    "indexing": {
                        "method": "in_memory",
                        "method_params": {"in_memory": None},
                    }
                }
            )
        )
        assert config.ingestion.indexing.params_for("in_memory") == {}

    def test_smoke_yaml_parses(self):
        config = load_config(CONFIGS_DIR / "smoke.yaml", dotenv_path=None)
        assert config.inference.fusion.group_by == "doc"

    def test_company_yaml_parses_with_env(self, monkeypatch):
        monkeypatch.setenv("ES_URL", "http://localhost:9200")
        monkeypatch.setenv("COMPANY_EMBEDDING_API_KEY", "k1")
        monkeypatch.setenv("COMPANY_LLM_API_KEY", "k2")
        config = load_config(CONFIGS_DIR / "company.yaml", dotenv_path=None)
        assert config.ingestion.indexing.method == "elasticsearch"
        assert config.inference.reranking.methods() == ["similarity", "llm"]
        gateway = config.inference.generation.params_for("gateway_openai_compatible")
        assert gateway["api_key"] == "k2"
        assert "model" not in gateway

    def test_docs_yaml_parses_with_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k1")
        config = load_config(CONFIGS_DIR / "docs.yaml", dotenv_path=None)
        assert config.ingestion.import_.method == "local_file"
        assert config.ingestion.parsing.method == "auto"
        assert config.ingestion.indexing.params_for()["index"] == "modular-rag-docs"
        assert config.evaluation is None  # 內建評估集對自備文件沒有意義

    def test_condense_yaml_parses_with_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k1")
        config = load_config(CONFIGS_DIR / "condense.yaml", dotenv_path=None)
        assert config.inference.generate_answer is False
        assert config.inference.query_transformation.methods() == [
            "jargon_mapping",
            "llm_rewrite",
        ]
        retrieval = config.inference.retrieval.params_for("hybrid")
        assert retrieval["boost_k_factor"] == 3
        assert config.inference.reranking.methods() == [
            "similarity",
            "llm_fact_check",
        ]
        assert config.inference.fusion.top_k == 3

    def test_company_yaml_fails_loud_without_env(self, monkeypatch):
        monkeypatch.delenv("ES_URL", raising=False)
        monkeypatch.delenv("COMPANY_EMBEDDING_API_KEY", raising=False)
        monkeypatch.delenv("COMPANY_LLM_API_KEY", raising=False)
        from rag.errors import ConfigError

        with pytest.raises(ConfigError, match="未設定的環境變數"):
            load_config(CONFIGS_DIR / "company.yaml", dotenv_path=None)

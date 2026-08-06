"""端到端煙霧測試:configs/smoke.yaml 全離線跑完 ingest → query。

一次驗證:方法鏈、mock embedding、hybrid 檢索、LLM 拆解(腳本)、
LLM 重排(腳本)、fusion 按文件聚合、mock 生成與 prompt 可稽核性。
"""

from __future__ import annotations

from pathlib import Path

from rag.builder import build_pipelines
from rag.config import load_config

SMOKE_YAML = Path(__file__).resolve().parent.parent / "configs" / "smoke.yaml"


def test_smoke_config_full_run(corpus_dir):
    config = load_config(SMOKE_YAML, dotenv_path=None)
    config.ingestion.import_.method_params["local_file"]["input_dir"] = str(corpus_dir)

    pipelines = build_pipelines(config)
    ingestion_result = pipelines.run_ingestion()
    assert ingestion_result["writer"]["documents_written"] > 0

    result = pipelines.query("ＦＡＩＳＳ 與 Elasticsearch 各支援什麼檢索?")

    # 拆解:mock 腳本固定回兩個子查詢,各自獨立檢索與重排
    assert len(result["subquery_results"]) == 2
    for subquery_docs in result["subquery_results"]:
        assert subquery_docs, "每個子查詢都應有(重排後的)結果"

    # fusion:group_by doc → group_key 是 doc_id,帶聚合診斷 metadata
    documents = result["documents"]
    assert 0 < len(documents) <= 3
    for doc in documents:
        assert doc.meta["group_key"] in {"faiss.txt", "sub/es.txt"}
        assert doc.meta["num_merged"] >= 1
        assert doc.meta["sources"][0]["rank"] >= 1
    assert [d.score for d in documents] == sorted(
        (d.score for d in documents), reverse=True
    ), "融合結果必須降冪"

    # prompt 可稽核:含 [chunk_id] 前綴與原始問題
    assert "::chunk_" in result["prompt"]
    assert "ＦＡＩＳＳ 與 Elasticsearch 各支援什麼檢索?" in result["prompt"]

    # mock 生成:可辨識的假答案,meta 帶模型資訊
    assert result["answer"].startswith("[mock 回答]")
    assert result["reply_meta"]["model"] == "mock"

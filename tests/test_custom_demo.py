"""configs/custom_demo.yaml + examples/custom_modules/ 的端到端回歸(五個槽位)。

這組範例是使用者帶去公司整合的骨架 —— 範例檔或示範 config 壞掉時
本測試即紅,保證骨架永遠可跑。載入的是 repo 中**真正的**檔案,
不做複本(否則測不到)。
"""

from __future__ import annotations

from pathlib import Path

from rag.builder import build_pipelines
from rag.config import expand_env_vars, parse_config, load_raw_config
from rag.trace import format_query_trace

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_CONFIG = REPO_ROOT / "configs" / "custom_demo.yaml"


def load_demo_config(corpus_dir):
    """載入真正的 custom_demo.yaml,僅把語料改指向 tmp corpus。"""
    raw = load_raw_config(DEMO_CONFIG)
    raw["ingestion"]["import"]["method_params"]["local_file"]["input_dir"] = str(
        corpus_dir
    )
    return parse_config(expand_env_vars(raw, source=str(DEMO_CONFIG)))


def test_custom_demo_end_to_end(corpus_dir, monkeypatch):
    # file 路徑相對執行目錄:與 run_demo 的使用方式一致(repo 根執行)
    monkeypatch.chdir(REPO_ROOT)
    pipelines = load_and_build(corpus_dir)
    pipelines.run_ingestion()

    result = pipelines.query("請假規則怎麼申請?")

    # custom retrieval:公司欄位已映射進 Document(content / meta)
    documents = result["documents"]
    assert documents, "custom retrieval 應回傳模擬語料"
    assert all("dockey" in doc.meta for doc in documents)
    assert all(doc.meta["chunk_id"].startswith(doc.meta["dockey"]) for doc in documents)

    # custom reranking:top_k=3 且原檢索分數保留在 meta
    assert len(documents) <= 3
    assert all("retrieval_score" in doc.meta for doc in documents)
    top = documents[0]
    assert "請假" in (top.content or ""), "重疊率重排應把請假規則排最前"

    # custom routing:獨立支線,結果附加於輸出
    route = result["routing"]
    assert route["category"] == "人資/差勤"
    assert "請假" in route["matched_keywords"]

    # custom generation:骨架的離線回覆 + meta 帶出來
    assert result["answer"].startswith("[公司 LLM 骨架回答]")
    assert result["reply_meta"]["model"] == "company-llm-v1"
    assert result["reply_meta"]["offline"] is True
    # prompt 仍由框架組:YAML 的 system_prompt 有進去
    assert "你是公司內部知識庫的助理" in result["prompt"]

    # custom fusion:一律執行(applied 由骨架回報)且 RRF 標記進 meta
    fusion = pipelines.inference.get_component("fusion")
    assert type(fusion).__name__ == "CompanyFusion"
    assert result["trace"]["fusion"]["applied"] is True
    assert all("num_merged" in doc.meta for doc in documents)

    # custom parsing(鏈中文件處理器):cleaner 掛在 parser 之後
    parser_2 = pipelines.ingestion.get_component("parser_2")
    assert type(parser_2).__name__ == "CompanyBoilerplateCleaner"

    # custom formatter:公司信封進 output 鍵,欄位來自 retriever 塞的 meta
    envelope = result["output"]
    assert envelope["question"] == "請假規則怎麼申請?"
    assert envelope["totalCount"] == len(documents)
    assert all(row["dockey"] for row in envelope["returnData"])

    lines = format_query_trace(result["trace"])
    joined = "\n".join(lines)
    assert "Routing" in joined
    assert "人資/差勤" in joined


def test_custom_demo_query_transformation_chain(corpus_dir, monkeypatch):
    """[normalize, custom] 方法鏈:詞庫擴充 + 依連接詞拆成多條子查詢。"""
    monkeypatch.chdir(REPO_ROOT)
    pipelines = load_and_build(corpus_dir)
    pipelines.run_ingestion()

    result = pipelines.query("請假規則以及報帳流程?")

    queries = result["trace"]["transforms"][-1]["queries"]
    assert len(queries) == 2, "「以及」應拆成兩條子查詢"
    assert any("差勤申請" in q for q in queries), "請假 應被詞庫擴充"
    assert any("費用核銷" in q for q in queries), "報帳 應被詞庫擴充"
    assert len(result["subquery_results"]) == 2


def test_custom_demo_runs_inference_only(corpus_dir, monkeypatch):
    """custom retrieval 不吃本地索引 → 這份 config 支援 --stage inference。"""
    monkeypatch.chdir(REPO_ROOT)
    pipelines = build_pipelines(load_demo_config(corpus_dir), stage="inference")

    assert pipelines.ingestion is None
    result = pipelines.query("請假規則怎麼申請?")
    assert result["documents"], "custom retrieval 不需要先跑 ingestion"
    assert result["answer"] is not None


def load_and_build(corpus_dir):
    return build_pipelines(load_demo_config(corpus_dir))


def test_custom_demo_routing_default_category(corpus_dir, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    pipelines = load_and_build(corpus_dir)
    pipelines.run_ingestion()
    result = pipelines.query("量子力學是什麼?")
    assert result["routing"]["category"] == "其他"

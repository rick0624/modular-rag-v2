"""configs/custom_demo.yaml + examples/custom_modules/ 的端到端回歸。

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
    raw["ingestion"]["import"]["params"]["input_dir"] = str(corpus_dir)
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

    # 生成與 trace 排版都可用
    assert result["answer"] is not None
    lines = format_query_trace(result["trace"])
    joined = "\n".join(lines)
    assert "Routing" in joined
    assert "人資/差勤" in joined


def load_and_build(corpus_dir):
    return build_pipelines(load_demo_config(corpus_dir))


def test_custom_demo_routing_default_category(corpus_dir, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    pipelines = load_and_build(corpus_dir)
    pipelines.run_ingestion()
    result = pipelines.query("量子力學是什麼?")
    assert result["routing"]["category"] == "其他"

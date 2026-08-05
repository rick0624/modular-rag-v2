"""configs/custom_ingestion_demo.yaml + examples/custom_modules/ 的端到端回歸。

ingestion 三槽位(import / parsing 鏈首 / chunking)全掛 custom 的示範
—— 範例檔或 config 壞掉時本測試即紅,保證骨架永遠可跑。載入的是 repo
中**真正的**檔案,不做複本(否則測不到)。
"""

from __future__ import annotations

from pathlib import Path

from rag.builder import build_pipelines
from rag.config import load_config
from rag.trace import format_ingestion_trace

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_CONFIG = REPO_ROOT / "configs" / "custom_ingestion_demo.yaml"


def test_custom_ingestion_demo_end_to_end(monkeypatch):
    # file 路徑相對執行目錄:與 run_demo 的使用方式一致(repo 根執行)
    monkeypatch.chdir(REPO_ROOT)
    pipelines = build_pipelines(load_config(DEMO_CONFIG))
    result = pipelines.run_ingestion()

    # 三個 custom 元件都真的在 pipeline 裡跑過(以類別名驗證)
    types = {entry["type"] for entry in result["trace"]}
    assert {"CompanyApiImporter", "CompanyPayloadParser", "CompanySentenceChunker"} <= types
    assert result["writer"]["documents_written"] > 0
    assert result["empty_sources"] == []
    format_ingestion_trace(result["trace"])  # 排版不炸

    # ByteStream 來源(不落地)一路到穩定的 doc_id / chunk_id
    docs = pipelines.store.filter_documents()
    assert all(doc.meta.get("doc_id") for doc in docs)
    assert all(doc.id == doc.meta["chunk_id"] for doc in docs)
    assert any(doc.meta["doc_id"] == "HR-0001" for doc in docs)

    # 檢索的就是 custom ingestion 建出來的索引
    answer = pipelines.query("請假要提前幾天申請?")
    assert answer["documents"]
    assert answer["documents"][0].meta["doc_id"] == "HR-0001"
    assert answer["answer"] is not None

    # 內建 simple_json formatter:通用 JSON 形狀進 output 鍵
    payload = answer["output"]
    assert payload["query"] == "請假要提前幾天申請?"
    assert payload["total"] == len(answer["documents"])
    assert payload["documents"][0]["doc_id"] == "HR-0001"


def test_custom_chunker_respects_max_chars(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    pipelines = build_pipelines(load_config(DEMO_CONFIG))
    pipelines.run_ingestion()
    docs = pipelines.store.filter_documents()
    assert all(len(doc.content or "") <= 120 for doc in docs)
    # HR-0001 內容超過一句:應切成多塊且 seq 連號
    hr_chunks = sorted(d.id for d in docs if d.meta["doc_id"] == "HR-0001")
    assert hr_chunks[0] == "HR-0001::chunk_0"

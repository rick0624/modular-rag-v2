"""逐步紀錄(trace)測試:ingestion 與 query 的每一步都要留下可稽核的輸出。

trace 是 CLI ``--trace`` 的資料來源;這裡驗證形狀與內容,不驗證印法。
"""

from __future__ import annotations

from conftest import make_config

from rag.builder import build_pipelines
from rag.config import parse_config


def _traceable_config(corpus_dir):
    """全離線的 condense 型配置:transform 鏈 + hybrid + 重排鏈。"""
    data = make_config(
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
                    "jargon_mapping": {"mapping": {"名次融合": "RRF"}},
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
            "retrieval": {"method": "hybrid", "params": {"top_k": 2}},
            "reranking": {
                "method": ["llm_fact_check"],
                "method_params": {
                    "llm_fact_check": {
                        "generator": {
                            "method": "mock",
                            "params": {"replies": ['{"relevant_indices": [1]}']},
                        }
                    }
                },
            },
            "generate_answer": False,
        },
    )
    del data["inference"]["generation"]
    return parse_config(data)


def test_ingestion_trace_lists_every_step(corpus_dir):
    pipelines = build_pipelines(_traceable_config(corpus_dir))
    result = pipelines.run_ingestion()

    names = [entry["component"] for entry in result["trace"]]
    assert names == ["importer", "parser", "chunker", "stamper", "embedder", "writer"]
    for entry in result["trace"]:
        assert entry["type"]  # 元件類別名,讓紀錄看得出實際用了哪個實作
    counts = {entry["component"]: entry["count"] for entry in result["trace"]}
    assert counts["importer"] > 0
    assert counts["chunker"] >= counts["parser"]  # 切塊只會變多不會變少
    assert result["writer"]["documents_written"] == counts["embedder"]


def test_query_trace_records_transforms_and_inner_steps(corpus_dir):
    pipelines = build_pipelines(_traceable_config(corpus_dir))
    pipelines.run_ingestion()
    result = pipelines.query("混合檢索的名次融合怎麼運作?")
    trace = result["trace"]

    # transform 鏈:每一段的產出查詢都留下,看得出改寫前後
    assert trace["query"] == "混合檢索的名次融合怎麼運作?"
    assert [entry["component"] for entry in trace["transforms"]] == [
        "transform",
        "transform_2",
    ]
    assert "RRF" in trace["transforms"][0]["queries"][0]  # jargon 替換後
    assert trace["transforms"][1]["queries"] == ["混合檢索 RRF 融合"]  # LLM 改寫後

    # 內部 pipeline 不再是黑箱:各路檢索與每段重排都有自己的一步
    assert len(trace["subqueries"]) == 1
    steps = trace["subqueries"][0]["steps"]
    names = [step["component"] for step in steps]
    assert "joiner" in names
    assert names[-1] == "ranker"  # 重排鏈的最後一段即 stage 輸出
    assert trace["subqueries"][0]["query"] == "混合檢索 RRF 融合"
    for step in steps:
        assert isinstance(step["documents"], list)

    # 事實查核只留 1 筆:過濾效果在 trace 中可直接對照
    assert len(steps[-1]["documents"]) == 1
    assert len(steps[-2]["documents"]) >= len(steps[-1]["documents"])

    # 檢索-only:fusion 是最後一步,沒有生成
    assert trace["fusion"]["documents"] == result["documents"]
    assert trace["generation"] is None


def test_fusion_trace_marks_passthrough(corpus_dir):
    """未設定 fusion + 單一查詢:元件仍在 pipeline 中,但要標示為未啟用。

    fusion 元件是無條件加進 pipeline 的,穿透時照樣出現在 trace;
    沒有 applied 旗標的話,紀錄會讓人誤以為結果被去重 / 重排過。
    """
    pipelines = build_pipelines(_traceable_config(corpus_dir))
    pipelines.run_ingestion()
    result = pipelines.query("混合檢索的名次融合怎麼運作?")

    assert result["trace"]["fusion"]["applied"] is False
    # 穿透 = 原樣輸出重排鏈的結果,連 group_key 都不會被加上
    last_step = result["trace"]["subqueries"][0]["steps"][-1]
    assert result["documents"] == last_step["documents"]
    assert all("group_key" not in doc.meta for doc in result["documents"])


def test_fusion_trace_marks_applied(corpus_dir):
    """config 有 fusion 區塊時,單一查詢也走融合,applied 為 True。"""
    data = make_config(
        ingestion={
            "import": {
                "method": "local_file",
                "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
            },
        },
        inference={"fusion": {"group_by": "doc", "strategy": "rrf", "top_k": 3}},
    )
    pipelines = build_pipelines(parse_config(data))
    pipelines.run_ingestion()
    result = pipelines.query("FAISS 支援哪些索引結構?")

    assert result["trace"]["fusion"]["applied"] is True
    assert all("group_key" in doc.meta for doc in result["documents"])


def test_query_trace_records_generation(corpus_dir):
    """開啟生成時,trace 另存實際 prompt、答案與模型 meta。"""
    data = make_config(
        ingestion={
            "import": {
                "method": "local_file",
                "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
            },
        },
        inference={
            "generation": {
                "method": "mock",
                "params": {"replies": ["FAISS 支援 Flat、IVF 與 HNSW。"]},
            },
        },
    )
    pipelines = build_pipelines(parse_config(data))
    pipelines.run_ingestion()
    result = pipelines.query("FAISS 支援哪些索引結構?")

    generation = result["trace"]["generation"]
    assert generation["answer"] == "FAISS 支援 Flat、IVF 與 HNSW。"
    assert "FAISS 支援哪些索引結構?" in generation["prompt"]
    assert generation["meta"] == result["reply_meta"]


def test_custom_fusion_without_applied_renders_custom_branch(corpus_dir, tmp_path):
    """custom fusion 未回報 applied:trace 為 None,排版走 custom 分支不炸。"""
    (tmp_path / "fusion.py").write_text(
        "from typing import Any\n"
        "from haystack import component\n"
        "from haystack.dataclasses import Document\n\n\n"
        "@component\n"
        "class SilentFusion:\n"
        "    @component.output_types(documents=list[Document])\n"
        "    def run(self, results: list[list[Document]]) -> dict[str, Any]:\n"
        "        return {'documents': [d for docs in results for d in docs]}\n",
        encoding="utf-8",
    )
    data = make_config(
        ingestion={
            "import": {
                "method": "local_file",
                "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
            },
        },
    )
    data["inference"]["fusion"] = {
        "method": "custom",
        "params": {"file": str(tmp_path / "fusion.py"), "class": "SilentFusion"},
    }
    pipelines = build_pipelines(parse_config(data))
    pipelines.run_ingestion()
    result = pipelines.query("FAISS 支援哪些索引結構?")

    assert result["trace"]["fusion"]["applied"] is None

    from rag.trace import format_query_trace

    joined = "\n".join(format_query_trace(result["trace"]))
    assert "Fusion(custom 元件)" in joined
    assert "未回報 applied" in joined
    # 既有兩種分支的字樣不可誤現
    assert "未啟用:原樣通過" not in joined


def test_formatter_trace_renders_and_absent_when_not_mounted(corpus_dir):
    data = make_config(
        ingestion={
            "import": {
                "method": "local_file",
                "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
            },
        },
        inference={"reranking": {"method": "none"}},
    )
    data["inference"]["formatter"] = {"method": "simple_json"}
    pipelines = build_pipelines(parse_config(data))
    pipelines.run_ingestion()
    result = pipelines.query("FAISS 支援哪些索引結構?")

    from rag.trace import format_query_trace

    joined = "\n".join(format_query_trace(result["trace"]))
    assert "Formatter(對外格式;SimpleJsonFormatter)" in joined
    # 終端機截斷,log 檔(limit=0)全印
    full = "\n".join(format_query_trace(result["trace"], limit=0))
    assert len(full) >= len(joined)

    # 未掛 formatter 的舊 trace dict(無此鍵)也要能排版
    legacy = dict(result["trace"])
    del legacy["formatter"]
    format_query_trace(legacy)

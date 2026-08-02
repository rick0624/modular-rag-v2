"""服務模式:FastAPI HTTP API(單一 process,ingestion + inference 同駐)。

一份 YAML config 啟動一個 KB 服務。兩個邏輯服務共用同一個 process 與
document store,結構上保證 embedding 等跨階段設定一致(inference 的查
詢端 embedder 本來就是從 ``ingestion.embedding`` 派生的)。

核心不變量:**啟動後只允許 inference / evaluation 區塊變更**。
ingestion 區塊(import / parsing / chunking / embedding / indexing)
決定索引裡的內容,改了就必須重建索引 —— 由 :mod:`rag.kb_meta` 的
指紋機制強制:

- ``POST /reload``:重讀 YAML,只重建 inference pipeline(store 沿用)。
  ingestion 指紋不符 → **409**,並導向 ``POST /ingest``。
- ``POST /ingest``:ingestion 設定變更的唯一正道。重讀 YAML、全量重建
  兩條 pipeline(全新 store)、重跑 ingestion、更新指紋。

併發:Haystack pipeline 不是 thread-safe,所有 run 與狀態切換都在同一
把 ``threading.Lock`` 內(查詢會被進行中的 /ingest 擋住)。因此
**uvicorn workers 必須為 1**(in-process store 與狀態無法跨 process)。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rag.builder import RagPipelines, build_inference_pipeline, build_pipelines
from rag.config import RAGConfig, load_config, load_raw_config
from rag.errors import ConfigError, RAGFrameworkError
from rag.kb_meta import (
    indexing_info,
    ingestion_fingerprint,
    read_fingerprint,
    write_fingerprint,
)
from rag.trace import format_query_trace

logger = logging.getLogger(__name__)


# 請求 / 回應模型必須在模組層級:`from __future__ import annotations` 下
# FastAPI 以字串型別註解 + get_type_hints 解析 endpoint 簽名,函式區域的
# 類別解析不到,body 參數會被誤判成 query 參數(422)。


class QueryRequest(BaseModel):
    query: str


class DocumentOut(BaseModel):
    chunk_id: str | None
    doc_id: str | None
    page: int | None
    score: float | None
    content: str | None


class QueryResponse(BaseModel):
    answer: str | None
    documents: list[DocumentOut]
    subquery_count: int
    prompt: str | None


class IngestStep(BaseModel):
    component: str
    type: str
    count: int | None


class IngestResponse(BaseModel):
    documents_written: int | None
    skipped_unchanged: int | None  # 增量:跳過的未變更切片數
    skipped_files: list[str]  # 檔案層增量:連 parse 都跳過的檔案
    empty_sources: list[str]  # 沒有產出任何切片的檔案(掃描檔需 OCR?)
    fingerprint: str
    steps: list[IngestStep]


class ReloadResponse(BaseModel):
    status: str
    fingerprint: str


class HealthResponse(BaseModel):
    status: str
    config_path: str
    indexing_method: str
    fingerprint: str


@dataclass
class ServiceState:
    """服務的可變狀態(swap 一律在 lock 內、以整個物件為單位)。"""

    config_path: Path
    pipelines: RagPipelines
    fingerprint: str
    lock: threading.Lock


_indexing_info = indexing_info  # 與 kb_meta 共用同一實作


def _run_ingest_and_stamp(
    pipelines: RagPipelines, config: RAGConfig, fingerprint: str
) -> dict[str, Any]:
    """跑 ingestion 並在成功後寫入指紋(先跑後寫:失敗不留新指紋)。"""
    result = pipelines.run_ingestion()
    method, index_name = _indexing_info(config)
    write_fingerprint(method, pipelines.store, fingerprint, index_name)
    return result


def create_app(config_path: str | Path, *, skip_ingest: bool = False) -> Any:
    """建立 FastAPI app(建構期就載入 config 並建 pipeline,錯誤 fail-fast)。

    Args:
        config_path: YAML 設定檔路徑;之後 /ingest 與 /reload 都重讀此檔。
        skip_ingest: 啟動時跳過 ingestion(索引已建好時用)。此時會比對
            索引上的指紋,不符或缺失(elasticsearch)→ 拒絕啟動。

    Raises:
        MissingDependencyError: 未安裝 fastapi(``pip install -e ".[service]"``)。
        ConfigError: 設定不合法,或 --skip-ingest 時指紋不符。
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.requests import Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        from rag.errors import MissingDependencyError

        raise MissingDependencyError(
            "fastapi", "服務模式"
        ) from exc

    path = Path(config_path)
    raw = load_raw_config(path)
    config = load_config(path)
    fingerprint = ingestion_fingerprint(raw)
    pipelines = build_pipelines(config)

    if skip_ingest:
        method, index_name = _indexing_info(config)
        stored = read_fingerprint(method, pipelines.store, index_name)
        if method == "elasticsearch" and stored != fingerprint:
            raise ConfigError(
                f"--skip-ingest 但索引 '{index_name}' 上的 ingestion 指紋"
                f"({stored or '不存在'})與目前設定({fingerprint[:12]}…)不符。"
                "索引內容可能與設定不一致;請移除 --skip-ingest 重新 ingest,"
                "或啟動後呼叫 POST /ingest"
            )
        if method != "elasticsearch":
            logger.warning(
                "--skip-ingest 搭配 %s 索引:索引是空的,查詢不會有結果;"
                "請呼叫 POST /ingest 建立內容", method or "in_memory",
            )
        logger.info("跳過啟動時 ingestion(--skip-ingest)")
        ingest_summary: dict[str, Any] | None = None
    else:
        result = _run_ingest_and_stamp(pipelines, config, fingerprint)
        ingest_summary = {"documents_written": _documents_written(result)}
        logger.info("啟動 ingestion 完成:%s", ingest_summary)

    state = ServiceState(
        config_path=path,
        pipelines=pipelines,
        fingerprint=fingerprint,
        lock=threading.Lock(),
    )
    logger.info("服務就緒:config=%s fingerprint=%s", path, fingerprint[:12])

    app = FastAPI(title="modular-rag-v2", description=__doc__)
    app.state.service = state  # 測試與外部檢視用

    @app.exception_handler(RAGFrameworkError)
    async def _framework_error(request: Request, exc: RAGFrameworkError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> Any:
        method, _ = _indexing_info(state.pipelines.config)
        return HealthResponse(
            status="ok",
            config_path=str(state.config_path),
            indexing_method=method,
            fingerprint=state.fingerprint,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> Any:
        with state.lock:
            result = state.pipelines.query(request.query)
        logger.info(
            "查詢 %r\n%s", request.query,
            "\n".join(format_query_trace(result["trace"], limit=0)),
        )
        return QueryResponse(
            answer=result["answer"],
            documents=[
                DocumentOut(
                    chunk_id=doc.meta.get("chunk_id"),
                    doc_id=doc.meta.get("doc_id"),
                    page=doc.meta.get("page"),
                    score=doc.score,
                    content=doc.content,
                )
                for doc in result["documents"]
            ],
            subquery_count=len(result["subquery_results"]),
            prompt=result["prompt"],
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest() -> Any:
        """全量重建索引 —— **ingestion 設定變更的唯一正道**。

        重讀 YAML(接受任何 ingestion 變更)、全新 store、兩條 pipeline
        全重建、重跑 ingestion、更新指紋。
        """
        new_raw = load_raw_config(state.config_path)
        new_config = load_config(state.config_path)
        new_fingerprint = ingestion_fingerprint(new_raw)
        with state.lock:
            new_pipelines = build_pipelines(new_config)
            result = _run_ingest_and_stamp(new_pipelines, new_config, new_fingerprint)
            state.pipelines = new_pipelines
            state.fingerprint = new_fingerprint
        logger.info("重新 ingestion 完成:fingerprint=%s", new_fingerprint[:12])
        return IngestResponse(
            documents_written=_documents_written(result),
            skipped_unchanged=(result.get("change_filter") or {}).get("skipped"),
            skipped_files=(result.get("source_filter") or {}).get("skipped_files") or [],
            empty_sources=result.get("empty_sources") or [],
            fingerprint=new_fingerprint,
            steps=[
                IngestStep(component=s["component"], type=s["type"], count=s["count"])
                for s in result["trace"]
            ],
        )

    @app.post("/reload", response_model=ReloadResponse)
    def reload() -> Any:
        """重讀 YAML,只重建 inference(store 沿用;ingestion 不可變)。"""
        new_raw = load_raw_config(state.config_path)
        new_fingerprint = ingestion_fingerprint(new_raw)
        if new_fingerprint != state.fingerprint:
            raise HTTPException(
                status_code=409,
                detail=(
                    "ingestion 區塊已變更(指紋不符),/reload 只允許 "
                    "inference / evaluation 變更。ingestion 變更會使索引內容"
                    "與設定不一致,請改用 POST /ingest 重建索引"
                ),
            )
        new_config = load_config(state.config_path)
        with state.lock:
            inference, meta = build_inference_pipeline(
                new_config, store=state.pipelines.store
            )
            old = state.pipelines
            state.pipelines = RagPipelines(
                config=new_config,
                ingestion=old.ingestion,
                inference=inference,
                store=old.store,
                query_entry=meta["query_entry"],
                generate_answer=meta["generate_answer"],
                transform_names=meta["transform_names"],
            )
        logger.info("inference 設定重載完成")
        return ReloadResponse(status="reloaded", fingerprint=state.fingerprint)

    return app


def _documents_written(ingestion_result: dict[str, Any]) -> int | None:
    return (ingestion_result.get("writer") or {}).get("documents_written")

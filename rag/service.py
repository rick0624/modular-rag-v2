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

階段選擇(``scripts/serve.py --stage``):``all``(預設)啟動時 ingest 一次
再開服務;``inference`` 跳過啟動 ingestion(索引已建好,比對指紋);
``ingestion`` 走 :func:`ingest_only` —— 只建索引、不開 port,供讀寫分離
部署的 writer 端(排程 / CI)使用。

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
    routing: dict[str, Any] | None = None  # 查詢分類;未設 routing 槽位時為 null
    # formatter 組出的對外格式;未設 formatter 槽位時為 null。
    # payload 走 HTTP 必須 JSON 可序列化(這是 wire 的限制,不是槽位的)。
    output: Any | None = None


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


def _run_ingest_and_stamp(
    pipelines: RagPipelines, config: RAGConfig, fingerprint: str
) -> dict[str, Any]:
    """跑 ingestion 並在成功後寫入指紋(先跑後寫:失敗不留新指紋)。"""
    result = pipelines.run_ingestion()
    method, index_name = indexing_info(config)
    write_fingerprint(method, pipelines.store, fingerprint, index_name)
    return result


def ingest_only(config_path: str | Path) -> dict[str, Any]:
    """只建索引:跑完 ingestion、寫指紋,不啟動服務(``--stage ingestion``)。

    讀寫分離部署的 writer 端 —— 由排程或 CI 呼叫把索引建好,查詢端再以
    ``create_app(..., stage="inference")`` 起來吃同一個索引(指紋就是兩者
    之間的握手:對不上就拒絕啟動)。因此只建 ingestion pipeline,
    inference 側元件(reranker 模型等)完全不載入。

    Returns:
        ``run_ingestion()`` 的結果,外加 ``fingerprint``。

    Raises:
        ConfigError: 設定不合法。
    """
    path = Path(config_path)
    raw = load_raw_config(path)
    config = load_config(path)
    fingerprint = ingestion_fingerprint(raw)
    pipelines = build_pipelines(config, stage="ingestion")
    result = _run_ingest_and_stamp(pipelines, config, fingerprint)
    result["fingerprint"] = fingerprint
    logger.info(
        "ingestion 完成(未啟動服務):documents_written=%s fingerprint=%s",
        _documents_written(result), fingerprint[:12],
    )
    return result


def create_app(config_path: str | Path, *, stage: str = "all") -> Any:
    """建立 FastAPI app(建構期就載入 config 並建 pipeline,錯誤 fail-fast)。

    Args:
        config_path: YAML 設定檔路徑;之後 /ingest 與 /reload 都重讀此檔。
        stage: ``"all"``(預設)啟動時跑一次 ingestion;``"inference"``
            跳過(索引已建好時用),此時會比對索引上的指紋,不符或缺失
            (elasticsearch)→ 拒絕啟動。只建索引不起服務請改用
            :func:`ingest_only`。

    Raises:
        MissingDependencyError: 未安裝 fastapi(``pip install -e ".[service]"``)。
        ConfigError: 設定不合法、``stage`` 不合法,或 stage="inference"
            時指紋不符。
    """
    if stage not in ("all", "inference"):
        raise ConfigError(
            f"create_app 的 stage 只能是 'all' 或 'inference'(收到 '{stage}');"
            "只建索引不起服務請用 rag.service.ingest_only()"
        )
    skip_ingest = stage == "inference"
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
    pipelines = build_pipelines(config, stage=stage)

    if skip_ingest:
        method, index_name = indexing_info(config)
        stored = read_fingerprint(method, pipelines.store, index_name)
        if method == "elasticsearch" and stored != fingerprint:
            raise ConfigError(
                f"--stage inference 但索引 '{index_name}' 上的 ingestion 指紋"
                f"({stored or '不存在'})與目前設定({fingerprint[:12]}…)不符。"
                "索引內容可能與設定不一致;請改用 --stage all 重新 ingest,"
                "或啟動後呼叫 POST /ingest"
            )
        if method != "elasticsearch":
            logger.warning(
                "--stage inference 搭配 %s 索引:索引是空的,查詢不會有結果;"
                "請呼叫 POST /ingest 建立內容", method or "in_memory",
            )
        logger.info("跳過啟動時 ingestion(--stage inference)")
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
        # 伺服器端也要留紀錄:只回 400 給 client 的話,server log 完全
        # 看不到這次請求為什麼被拒,事後無從追查。
        logger.warning(
            "請求被拒(%s %s):%s: %s",
            request.method, request.url.path, type(exc).__name__, exc,
        )
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> Any:
        method, _ = indexing_info(state.pipelines.config)
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
            routing=result.get("routing"),
            output=result.get("output"),
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
                routing_enabled=meta["routing_enabled"],
                formatter_enabled=meta["formatter_enabled"],
                stage=old.stage,  # /reload 不建 ingestion,沿用原本的建構意圖
            )
        logger.info("inference 設定重載完成")
        return ReloadResponse(status="reloaded", fingerprint=state.fingerprint)

    return app


def _documents_written(ingestion_result: dict[str, Any]) -> int | None:
    return (ingestion_result.get("writer") or {}).get("documents_written")

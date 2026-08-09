#!/usr/bin/env python
"""服務模式:以一份 YAML config 啟動 KB 的 HTTP API(FastAPI)。

    pip install -e ".[service]"
    python scripts/serve.py --config configs/docs.yaml
    python scripts/serve.py --config configs/docs.yaml --stage inference # 索引已建好
    python scripts/serve.py --config configs/docs.yaml --stage ingestion # 只建索引就結束

``--stage`` 決定這個 process 做哪一段(預設 ``all``:啟動時 ingest 一次
再開服務):

- ``ingestion``:只建索引、寫指紋,**不開 port**(建完就結束)。讀寫分離
  部署的 writer 端 —— 排程 / CI 建索引,查詢服務另外用 ``--stage inference``
  起來吃同一個索引(指紋就是兩者之間的握手)。
- ``inference``:跳過啟動時的 ingestion,索引沿用既有內容(elasticsearch
  會比對 ingestion 指紋,不符則拒絕啟動)。

端點:
    GET  /health   狀態(config、indexing 方法、ingestion 指紋)
    POST /query    {"query": "..."} → 檢索結果(+答案,若有開生成)
    POST /reload   重讀 YAML,只重建 inference;ingestion 段變更 → 409
    POST /ingest   全量重建索引(ingestion 設定變更的唯一正道)

批次執行(一次性 ingest → 查詢 → 評估)請用 scripts/run_demo.py
(同樣支援 ``--stage``)。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.logging_config import (  # noqa: E402
    default_log_path,
    quiet_dependency_handlers,
    setup_logging,
    warning_tally,
)
from sample_data import ensure_sample_data  # noqa: E402
from rag.service import create_app, ingest_only  # noqa: E402

logger = logging.getLogger("rag.serve")


def main() -> None:
    parser = argparse.ArgumentParser(description="modular-rag-v2 HTTP 服務")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="YAML 設定檔路徑"
    )
    parser.add_argument("--host", default="127.0.0.1", help="綁定位址")
    parser.add_argument("--port", type=int, default=8000, help="埠號")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "ingestion", "inference"],
        help="這個 process 做哪一段:all=啟動時 ingest 一次再開服務(預設);"
        "ingestion=只建索引就結束(不開 port);"
        "inference=跳過啟動 ingestion(索引已建好,會比對 ingestion 指紋)",
    )
    parser.add_argument(
        "--log-file", default=None, help="log 檔路徑(預設 logs/serve-<時間戳>.log)"
    )
    parser.add_argument("--no-log-file", action="store_true", help="不寫 log 檔")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="終端機的日誌層級(log 檔一律 DEBUG;服務模式預設 INFO)",
    )
    args = parser.parse_args()

    # tqdm 進度條不走 logging,只能靠環境變數關;要在 ML 套件載入前設定。
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    log_file = None if args.no_log_file else (
        args.log_file or default_log_path().with_name(
            default_log_path().name.replace("run-", "serve-")
        )
    )
    log_path = setup_logging(log_file, console_level=args.log_level)
    if log_path is not None:
        print(f"完整紀錄:{log_path}")

    stage = args.stage

    ensure_sample_data()  # 範例語料與評估集(已存在的檔案不覆寫)

    if stage == "ingestion":
        # 只建索引:不開 port,建完就結束(讀寫分離部署的 writer 端)。
        try:
            result = ingest_only(args.config)
        except Exception:
            # 例外預設只進 stderr —— 補進 log 檔,排程 / CI 事後可查。
            logger.exception("ingestion 失敗(未捕捉的例外),traceback 如下")
            raise
        quiet_dependency_handlers()
        written = (result.get("writer") or {}).get("documents_written") or 0
        print(f"已索引 {written} 個切片({args.config})")
        print(f"ingestion 指紋:{result['fingerprint']}")
        empty_sources = result.get("empty_sources")
        if empty_sources:
            print(
                f"⚠ 以下檔案沒有產出任何切片(掃描檔需要 OCR?詳見 log):"
                f"{'、'.join(empty_sources)}"
            )
        warnings = warning_tally()
        if warnings:
            print(
                f"⚠ 本次 ingestion 有 {len(warnings)} 則警告"
                "(可能含 fail-soft 降級),詳見 log"
            )
        print("(--stage ingestion:未啟動服務;查詢請以 --stage inference 起另一個 process)")
        return

    try:
        app = create_app(args.config, stage=stage)
    except Exception:
        logger.exception("服務啟動失敗(未捕捉的例外),traceback 如下")
        raise
    quiet_dependency_handlers()  # 建 pipeline 時才載入的套件(HF…)補一次

    import uvicorn

    # workers 必須 = 1:store 與服務狀態都在 process 內,查詢/重建以
    # lock 序列化(見 rag/service.py 模組 docstring)。
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""服務模式:以一份 YAML config 啟動 KB 的 HTTP API(FastAPI)。

    pip install -e ".[service]"
    python scripts/serve.py --config configs/docs.yaml
    python scripts/serve.py --config configs/docs.yaml --skip-ingest   # 索引已建好

端點:
    GET  /health   狀態(config、indexing 方法、ingestion 指紋)
    POST /query    {"query": "..."} → 檢索結果(+答案,若有開生成)
    POST /reload   重讀 YAML,只重建 inference;ingestion 段變更 → 409
    POST /ingest   全量重建索引(ingestion 設定變更的唯一正道)

批次執行(一次性 ingest → 查詢 → 評估)請用 scripts/run_demo.py。
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
)
from rag.sample_data import ensure_sample_data  # noqa: E402
from rag.service import create_app  # noqa: E402

logger = logging.getLogger("rag.serve")


def main() -> None:
    parser = argparse.ArgumentParser(description="modular-rag-v2 HTTP 服務")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="YAML 設定檔路徑"
    )
    parser.add_argument("--host", default="127.0.0.1", help="綁定位址")
    parser.add_argument("--port", type=int, default=8000, help="埠號")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="啟動時不跑 ingestion(索引已建好;會比對 ingestion 指紋)",
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

    ensure_sample_data()  # 範例語料與評估集(已存在的檔案不覆寫)
    app = create_app(args.config, skip_ingest=args.skip_ingest)
    quiet_dependency_handlers()  # 建 pipeline 時才載入的套件(HF…)補一次

    import uvicorn

    # workers 必須 = 1:store 與服務狀態都在 process 內,查詢/重建以
    # lock 序列化(見 rag/service.py 模組 docstring)。
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()

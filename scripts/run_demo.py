#!/usr/bin/env python
"""端到端 demo:ingestion → 查詢 → (選填)評估。

    python scripts/run_demo.py                                   # 離線,不需金鑰
    python scripts/run_demo.py --config configs/smoke.yaml \
        --query "混合檢索怎麼運作?"
    python scripts/run_demo.py --config configs/condense.yaml --trace
        # 終端機逐步印出每個元件做了什麼(改寫、各路檢索、每段重排、融合)

每次執行都會另外寫一份完整紀錄到 ``logs/run-<時間戳>.log``
(含 LLM 實際 prompt 與回覆、每步全部切片,不截斷);
``--log-file`` 可指定路徑,``--no-log-file`` 可關閉。

長駐服務(不必每問一題重跑 ingestion)請用 scripts/serve.py。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import build_pipelines, load_config  # noqa: E402
from rag.evaluation import run_evaluation  # noqa: E402
from rag.logging_config import (  # noqa: E402
    default_log_path,
    quiet_dependency_handlers,
    setup_logging,
)
from rag.sample_data import ensure_sample_data  # noqa: E402
from rag.trace import (  # noqa: E402
    snippet,
    format_ingestion_trace,
    format_query_trace,
)

# 名字掛在 rag.* 底下,才會跟著吃到 rag 的 DEBUG 層級與 log 檔 handler。
logger = logging.getLogger("rag.demo")


def _emit(lines: list[str], to_console: bool) -> None:
    """同一份內容:終端機(選擇性)與 log 檔(一律)。"""
    for line in lines:
        if to_console:
            print(line)
        logger.info(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="modular-rag-v2 端到端 demo")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="YAML 設定檔路徑"
    )
    parser.add_argument("--query", default="FAISS 支援哪些索引結構?", help="查詢文字")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="終端機也逐步印出每個元件的輸入輸出(log 檔一律完整記錄)",
    )
    parser.add_argument(
        "--trace-docs",
        type=int,
        default=5,
        help="--trace 時終端機每步最多印幾筆切片(0 = 全印;預設 5)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="log 檔路徑(預設 logs/run-<時間戳>.log)",
    )
    parser.add_argument(
        "--no-log-file", action="store_true", help="不寫 log 檔"
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="終端機的日誌層級(log 檔一律 DEBUG;預設 WARNING)",
    )
    args = parser.parse_args()

    # tqdm 進度條不走 logging,只能靠環境變數關;要在 ML 套件載入前設定。
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    log_file = None if args.no_log_file else (args.log_file or default_log_path())
    log_path = setup_logging(log_file, console_level=args.log_level)

    ensure_sample_data()  # 範例語料與評估集(已存在的檔案不覆寫)
    config = load_config(args.config)
    logger.info("=" * 70)
    logger.info("config=%s  query=%r", args.config, args.query)
    logger.info("=" * 70)
    pipelines = build_pipelines(config)
    quiet_dependency_handlers()  # 建 pipeline 時才載入的套件(HF…)補一次

    logger.info("=== Ingestion(%s)===", args.config)
    ingestion_result = pipelines.run_ingestion()
    _emit(format_ingestion_trace(ingestion_result["trace"]), args.trace)
    written = ingestion_result.get("writer", {}).get("documents_written") or 0
    notes = []
    skipped_chunks = ingestion_result.get("change_filter", {}).get("skipped")
    if skipped_chunks:
        notes.append(f"增量:{skipped_chunks} 筆切片未變更")
    skipped_files = ingestion_result.get("source_filter", {}).get("skipped_files")
    if skipped_files:
        notes.append(f"{len(skipped_files)} 個檔案未變更未重新解析")
    extra = f"({'; '.join(notes)})" if notes else ""
    print(f"已索引 {written} 個切片{extra}({args.config})")
    empty_sources = ingestion_result.get("empty_sources")
    if empty_sources:
        print(
            f"⚠ 以下檔案沒有產出任何切片(掃描檔需要 OCR?詳見 log):"
            f"{'、'.join(empty_sources)}"
        )

    logger.info("=== 查詢 === %s", args.query)
    result = pipelines.query(args.query)
    # log 檔用 limit=0(全印);終端機照 --trace-docs 截斷。
    logger.info("\n%s", "\n".join(format_query_trace(result["trace"], limit=0)))
    if args.trace:
        print()
        print("\n".join(format_query_trace(result["trace"], args.trace_docs)))

    subqueries = len(result["subquery_results"])
    extra = f",{subqueries} 個子查詢" if subqueries > 1 else ""
    print(f"\n查詢:{args.query}{extra}")
    if result.get("routing") is not None:
        print(f"查詢分類:{result['routing']}")
    if not result["documents"]:
        print("(無檢索結果 —— 逐步經過看 log 檔或加 --trace)")
    for rank, doc in enumerate(result["documents"], start=1):
        merged = doc.meta.get("num_merged", 1)
        tag = f" ×{merged}" if merged > 1 else ""  # 融合時併了幾筆
        print(
            f"{rank}. {doc.score:.4f}  [{doc.meta.get('chunk_id')}]{tag}  "
            f"{snippet(doc.content, 50)}"
        )
    if result["answer"] is None:
        print("(檢索-only,未生成答案)")
    else:
        print(f"\n{result['answer']}")
        logger.info("--- 送出的 prompt(完整)---\n%s", result["prompt"])
        logger.info("--- 回答 ---\n%s", result["answer"])
        logger.info("--- 生成 meta ---%s", result["reply_meta"])
        if args.trace:
            print("\n--- 送出的 prompt ---")
            print(result["prompt"])

    if config.evaluation is not None:
        logger.info("=== 評估 ===")
        evaluation = run_evaluation(pipelines, config)
        metrics = evaluation["metrics"]
        print(
            f"\n評估:hit_rate={metrics['hit_rate']:.3f}  "
            f"mrr={metrics['mrr']:.3f}({evaluation['num_cases']} 題)"
        )
        for row in evaluation["per_query"]:
            line = (
                f"  {'✓' if row['hit'] else '✗'} "
                f"rr={row['reciprocal_rank']:.3f}  {row['query']}"
            )
            logger.info(line)
            if args.trace:  # 逐題結果平時是雜訊,需要時看 log 檔或 --trace
                print(line)

    if log_path is not None:
        print(f"紀錄:{log_path}")


if __name__ == "__main__":
    main()

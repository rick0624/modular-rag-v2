#!/usr/bin/env python
"""端到端 demo:ingestion → 查詢 → (選填)評估。

    python scripts/run_demo.py                                   # 離線,不需金鑰
    python scripts/run_demo.py --config configs/smoke.yaml \
        --query "混合檢索怎麼運作?"
    python scripts/run_demo.py --config configs/condense.yaml --trace
        # 終端機逐步印出每個元件做了什麼(改寫、各路檢索、每段重排、融合)

``--stage`` 決定這次跑哪一段(預設 ``all``:ingestion → 查詢 → 評估):

    python scripts/run_demo.py --config configs/docs.yaml --stage ingestion
    python scripts/run_demo.py --config configs/docs.yaml --stage inference

- ``ingestion``:只建索引(建完寫 ingestion 指紋,serve.py --stage
  inference 起得來),不查詢也不評估。
- ``inference``:只查詢 + 評估,索引沿用既有內容。改 prompt / 改寫 /
  重排的迭代不必每次重新切塊與 embedding。適用於索引已建好
  (elasticsearch,會比對 ingestion 指紋)、或 retrieval 走 custom 外部
  檢索不吃本地索引時;in_memory 索引是空的,單獨用會查不到東西
  (會出聲提醒)。

每次執行都會另外寫一份完整紀錄到 ``logs/run-<時間戳>.log``
(含 LLM 實際 prompt 與回覆、每步全部切片,不截斷);
``--log-file`` 可指定路徑,``--no-log-file`` 可關閉。

長駐服務(不必每問一題重跑 ingestion)請用 scripts/serve.py
(同樣支援 ``--stage``)。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import build_pipelines, load_config  # noqa: E402
from rag.config import RAGConfig, load_raw_config  # noqa: E402
from rag.evaluation import run_evaluation  # noqa: E402
from rag.kb_meta import (  # noqa: E402
    indexing_info,
    ingestion_fingerprint,
    read_fingerprint,
    write_fingerprint,
)
from rag.logging_config import (  # noqa: E402
    default_log_path,
    quiet_dependency_handlers,
    setup_logging,
    warning_tally,
)
from sample_data import ensure_sample_data  # noqa: E402
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


def _warn(message: str) -> None:
    """終端機與 log 檔同時出聲(安靜地查一個空索引是最難查的問題)。

    與 :func:`_emit` 同機制:print 負責終端機(不受 ``--log-level``
    影響,一定看得到),log 走 INFO —— 用 warning 會被終端機 handler
    再印一次,同一句話出現兩行。
    """
    print(f"⚠ {message}")
    logger.info(message)


def _stamp_fingerprint(config_path: str, config: RAGConfig, store: object) -> None:
    """ingestion 成功後把設定指紋寫到索引旁(與服務模式同一機制)。

    這是 ``--stage ingestion`` 與後續 ``--stage inference``(或
    ``serve.py --stage inference``)之間的握手:指紋對得上,查詢端才敢
    相信索引內容就是這份 config 產出的。in_memory 的指紋隨 process 消失,
    寫了也不會有人讀到,但成本是零,不必為此分支。
    """
    method, index_name = indexing_info(config)
    if not method:  # escape hatch:原生 ingestion pipeline,無槽位資訊
        return
    try:
        write_fingerprint(
            method, store, ingestion_fingerprint(load_raw_config(config_path)), index_name
        )
    except Exception as exc:  # 寫不進去不該讓已完成的 ingestion 算失敗
        _warn(
            f"索引建好了,但 ingestion 指紋寫入失敗"
            f"({type(exc).__name__}: {exc});"
            "之後以 --stage inference 啟動會因為指紋對不上而被擋下"
        )


def _report_skipped_ingestion(
    config_path: str, config: RAGConfig, store: object
) -> None:
    """``--stage inference``:確認索引狀態,不一致就出聲(不中止)。

    比照服務模式:elasticsearch 比對 ingestion 指紋(設定變了但索引沒重建
    → 查詢結果會無聲劣化);in_memory 索引本來就是空的,除非 retrieval
    走 custom(自帶外部檢索後端,不吃本地索引)。
    """
    method, index_name = indexing_info(config)
    retrieval_method = (
        config.inference.retrieval.methods()[0]
        if config.inference is not None
        else ""
    )
    if method != "elasticsearch":
        if retrieval_method == "custom":
            print(f"跳過 ingestion(retrieval 走 custom,不吃本地 {method} 索引)")
        else:
            _warn(
                f"--stage inference 搭配 {method or 'in_memory'} 索引:索引是空的,"
                "查詢不會有結果。持久化索引請改用 elasticsearch,"
                "或改用 --stage all 重跑 ingestion"
            )
        return

    try:
        stored = read_fingerprint(method, store, index_name)
    except Exception as exc:  # 連線失敗 / 索引不存在以外的錯誤
        _warn(
            f"--stage inference 但讀不到索引 '{index_name}' 上的 ingestion 指紋"
            f"({type(exc).__name__}: {exc});無法確認索引與設定是否一致"
        )
        return
    current = ingestion_fingerprint(load_raw_config(config_path))
    if stored != current:
        _warn(
            f"--stage inference 但索引 '{index_name}' 上的 ingestion 指紋"
            f"({(stored or '不存在')[:12]}…)與目前設定({current[:12]}…)不符;"
            "索引內容可能與設定不一致(embedding 模型 / 切塊參數變了?)。"
            "要以目前設定重建請改用 --stage all"
        )
        return
    print(f"跳過 ingestion(索引 '{index_name}' 的 ingestion 指紋相符)")


def _report_warnings(log_path) -> None:
    """執行結束時總結本次警告:fail-soft 讓流程「看似成功」,這裡出聲。"""
    warnings = warning_tally()
    if not warnings:
        return
    print(
        f"\n⚠ 本次執行有 {len(warnings)} 則警告"
        "(可能含 fail-soft 降級 —— 流程完成但部分步驟已退化):"
    )
    for message in warnings[:5]:
        print(f"  - {message[:110]}")
    if len(warnings) > 5:
        print(f"  …其餘 {len(warnings) - 5} 則")
    if log_path is not None:
        print(f"完整內容見 log:{log_path}")


def _run(args, log_path) -> None:
    ensure_sample_data()  # 範例語料與評估集(已存在的檔案不覆寫)
    config = load_config(args.config)
    logger.info("=" * 70)
    logger.info(
        "config=%s  stage=%s  query=%r", args.config, args.stage, args.query
    )
    logger.info("=" * 70)
    pipelines = build_pipelines(config, stage=args.stage)
    quiet_dependency_handlers()  # 建 pipeline 時才載入的套件(HF…)補一次

    if args.stage == "inference":
        logger.info("=== 跳過 Ingestion(--stage inference)===")
        _report_skipped_ingestion(args.config, config, pipelines.store)
    else:
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
        _stamp_fingerprint(args.config, config, pipelines.store)

    if args.stage == "ingestion":
        # 只建索引:不查詢也不評估(查詢請用 --stage inference / all)
        if log_path is not None:
            print(f"紀錄:{log_path}")
        return

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


def main() -> None:
    parser = argparse.ArgumentParser(description="modular-rag-v2 端到端 demo")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="YAML 設定檔路徑"
    )
    parser.add_argument("--query", default="FAISS 支援哪些索引結構?", help="查詢文字")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "ingestion", "inference"],
        help="這次跑哪一段:all=ingestion → 查詢 → 評估(預設);"
        "ingestion=只建索引(並寫 ingestion 指紋);"
        "inference=只查詢 + 評估,索引沿用既有內容"
        "(elasticsearch 會比對 ingestion 指紋)",
    )
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

    try:
        _run(args, log_path)
    except Exception:
        # 例外預設只進 stderr,不進 log 檔 —— 這裡補上,事後查 log 檔
        # 才看得到這次執行是「失敗」而不是「沒跑完」。
        logger.exception("執行失敗(未捕捉的例外),traceback 如下")
        raise
    finally:
        _report_warnings(log_path)


if __name__ == "__main__":
    main()

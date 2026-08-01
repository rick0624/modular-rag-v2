#!/usr/bin/env python
"""端到端 demo:ingestion → 查詢 → (選填)評估。

    python scripts/run_demo.py                                   # 離線,不需金鑰
    python scripts/run_demo.py --config configs/smoke.yaml \
        --query "混合檢索怎麼運作?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import build_pipelines, load_config  # noqa: E402
from rag.evaluation import run_evaluation  # noqa: E402
from rag.sample_data import ensure_sample_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="modular-rag-v2 端到端 demo")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="YAML 設定檔路徑"
    )
    parser.add_argument("--query", default="FAISS 支援哪些索引結構?", help="查詢文字")
    args = parser.parse_args()

    ensure_sample_data()  # 範例語料與評估集(已存在的檔案不覆寫)
    config = load_config(args.config)
    pipelines = build_pipelines(config)

    print(f"=== Ingestion({args.config}) ===")
    ingestion_result = pipelines.run_ingestion()
    written = ingestion_result.get("writer", {}).get("documents_written")
    print(f"已寫入切片數:{written}")

    print(f"\n=== 查詢 ===\n{args.query}")
    result = pipelines.query(args.query)
    if len(result["subquery_results"]) > 1:
        print(f"(拆解為 {len(result['subquery_results'])} 個子查詢)")
    print("\n--- 檢索結果(融合後)---")
    for rank, doc in enumerate(result["documents"], start=1):
        extras = ""
        if "group_key" in doc.meta:
            extras = f"(group={doc.meta['group_key']}, merged={doc.meta['num_merged']})"
        head = " ".join((doc.content or "").split())[:60]
        print(f"{rank}. score={doc.score:.4f} [{doc.meta.get('chunk_id')}]{extras}")
        print(f"   {head}")
    if result["answer"] is None:
        print("\n(generate_answer: false — 僅檢索,略過生成)")
    else:
        print("\n--- 送出的 prompt(前 400 字)---")
        print(result["prompt"][:400])
        print("\n--- 回答 ---")
        print(result["answer"])

    if config.evaluation is not None:
        print("\n=== 評估 ===")
        evaluation = run_evaluation(pipelines, config)
        metrics = evaluation["metrics"]
        print(
            f"hit_rate={metrics['hit_rate']:.3f}  mrr={metrics['mrr']:.3f}"
            f"(共 {evaluation['num_cases']} 題)"
        )
        for row in evaluation["per_query"]:
            mark = "✓" if row["hit"] else "✗"
            print(f"  {mark} rr={row['reciprocal_rank']:.3f}  {row['query']}")


if __name__ == "__main__":
    main()

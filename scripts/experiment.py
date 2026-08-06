#!/usr/bin/env python
"""管線組合實驗:程式化生成 config、逐組合執行查詢、收集原始結果。

用法:改下面「實驗定義」區塊,然後:

    python scripts/experiment.py

或在你自己的評估腳本裡:

    from experiment import run_experiments
    records = run_experiments()
    for rec in records:
        ...  # rec["results"] 是 [(query, pipelines.query 的完整輸出), ...]

每筆 record 都帶出處:label(人讀的組合名)、overrides(這個組合改了
哪些槽位)、config(完整 config dict,可存檔重現)。評估邏輯不在此
腳本內 —— 拿 records 之後自己接。
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import build_pipelines  # noqa: E402
from rag.config import RAGConfig, load_raw_config  # noqa: E402

# 掃描時保持輸出乾淨;想看每個組合的內部 log 就把這行拿掉。
logging.disable(logging.WARNING)

# ─────────────────────────── 實驗定義(改這裡)───────────────────────────

BASE_CONFIG = "configs/default.yaml"

# "one_at_a_time":基線 + 每次只動一個槽位(每槽位每方法各跑一次)。
# "product":SLOT_OPTIONS 全交叉乘積。小網格 = 把 SLOT_OPTIONS 收窄到
#            2-3 個有交互作用的槽位後用 product,不需要第三種模式。
MODE = "one_at_a_time"

# 槽位 → 要比較的選項清單。選項的三種寫法:
#   字串   → 只覆蓋 method,參數沿用 default.yaml 的 method_params 型錄
#   list  → 方法鏈(如 ["normalize", "llm_decompose"])
#   dict  → 整個槽位配置直接替換(要自訂參數、或 fusion / routing 這類
#           基底沒啟用的槽位時用),如:
#           {"method": "llm", "params": {"top_k": 3}}
SLOT_OPTIONS: dict[str, list[Any]] = {
    "inference.retrieval": ["bm25", "embedding", "hybrid"],
    "inference.query_transformation": ["passthrough", "normalize"],
}

QUERIES = [
    "FAISS 支援哪些索引結構?",
    "Elasticsearch 的用途是什麼?",
]

# ─────────────────────────── 組合生成 ───────────────────────────


def apply_option(cfg: dict, dotted_slot: str, option: Any) -> None:
    """把一個選項套進 config dict(就地修改)。"""
    section, slot = dotted_slot.split(".")
    if isinstance(option, dict):
        cfg[section][slot] = copy.deepcopy(option)
    else:  # 字串或方法鏈:只換 method,method_params 型錄原樣保留
        cfg[section].setdefault(slot, {})["method"] = option


def option_name(option: Any) -> str:
    """選項的簡短名稱(組 label 用)。"""
    if isinstance(option, dict):
        return str(option.get("method", "custom"))
    if isinstance(option, list):
        return "+".join(map(str, option))
    return str(option)


def make_variants(base: dict) -> list[tuple[str, dict, dict]]:
    """回傳 [(label, overrides, config_dict), ...]。"""
    variants = []
    if MODE == "one_at_a_time":
        variants.append(("baseline", {}, copy.deepcopy(base)))
        for dotted, options in SLOT_OPTIONS.items():
            slot = dotted.split(".")[1]
            for opt in options:
                cfg = copy.deepcopy(base)
                apply_option(cfg, dotted, opt)
                variants.append((f"{slot}={option_name(opt)}", {dotted: opt}, cfg))
    elif MODE == "product":
        slots = list(SLOT_OPTIONS)
        for combo in itertools.product(*(SLOT_OPTIONS[s] for s in slots)):
            cfg = copy.deepcopy(base)
            overrides = dict(zip(slots, combo))
            for dotted, opt in overrides.items():
                apply_option(cfg, dotted, opt)
            label = " + ".join(
                f"{d.split('.')[1]}={option_name(o)}" for d, o in overrides.items()
            )
            variants.append((label, overrides, cfg))
    else:
        raise ValueError(f"未知的 MODE:{MODE!r}(可用:one_at_a_time / product)")
    return variants


# ─────────────────────────── 執行 ───────────────────────────


def run_experiments() -> list[dict]:
    """逐組合建管線、跑查詢;ingestion 相同的組合共用同一個索引。"""
    base = load_raw_config(BASE_CONFIG)
    base.pop("evaluation", None)  # 評估自己接,不用 config 裡的 evaluation 區塊
    records: list[dict] = []
    stores: dict[str, Any] = {}  # ingestion 區塊 JSON → 已建好的索引

    for label, overrides, cfg_dict in make_variants(base):
        record = {"label": label, "overrides": overrides, "config": cfg_dict,
                  "results": [], "error": None}
        records.append(record)
        try:
            key = json.dumps(cfg_dict["ingestion"], sort_keys=True)
            config = RAGConfig.model_validate(cfg_dict)
            if key in stores:  # 索引沿用,只重建 inference 管線
                pipelines = build_pipelines(config, stage="inference",
                                            store=stores[key])
            else:
                pipelines = build_pipelines(config)
                pipelines.run_ingestion()
                stores[key] = pipelines.store
            record["results"] = [(q, pipelines.query(q)) for q in QUERIES]
        except Exception as exc:  # 一個壞組合不中斷整批
            record["error"] = f"{type(exc).__name__}: {exc}"
    return records


if __name__ == "__main__":
    for rec in run_experiments():
        if rec["error"]:
            print(f"[FAIL] {rec['label']}: {rec['error']}")
        else:
            docs = [len(r["documents"]) for _, r in rec["results"]]
            print(f"[OK]   {rec['label']}  各題檢回 {docs}")

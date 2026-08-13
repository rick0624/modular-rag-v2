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

需要多個槽位綁定一起換(例如 embedding 模型與 custom retrieval 必須
一致)時,用 bundle 選項 —— 見 SLOT_OPTIONS 的說明。
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

# 槽位 → 要比較的選項清單。選項的四種寫法:
#   字串   → 只覆蓋 method,參數沿用 default.yaml 的 method_params 型錄
#   list  → 方法鏈(如 ["normalize", "llm_decompose"])
#   dict  → 整個槽位配置直接替換(要自訂參數、或 fusion / routing 這類
#           基底沒啟用的槽位時用),如:
#           {"method": "llm", "params": {"top_k": 3}}
#   bundle → key 含「.」的 dict:一次覆蓋多個槽位,綁定一起換(例如
#           ingestion 的 embedding 模型跟 custom retrieval 用的模型必須
#           一致、或 retriever / fusion / formatter 要當一組 preset 時)。
#           此時外層 key 只是維度名(自取,不必是槽位路徑),product 模式
#           下整組作為一個維度參與交叉;"_label" 選填,當這個選項的名字:
#           "embedding_stack": [
#               {"_label": "openai-small",
#                "ingestion.embedding": {"method": "api_embedding", "params": {...}},
#                "inference.retrieval": {"method": "custom", "params": {...}}},
#               {"_label": "minilm-local", ...},
#           ]
#           同一個槽位只能屬於一個維度(重疊時 make_variants 直接報錯)。
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


def is_bundle(option: Any) -> bool:
    """bundle 選項 = key 含「.」的 dict(一次覆蓋多個槽位)。

    不會與「整個槽位配置」的 dict 寫法混淆:後者的 key 是 method /
    params 等欄位名,不含「.」。
    """
    return isinstance(option, dict) and any("." in key for key in option)


def resolve_overrides(dim: str, option: Any) -> dict[str, Any]:
    """把一個選項展成 {dotted_slot: 選項} 的覆蓋表。"""
    if is_bundle(option):
        return {k: v for k, v in option.items() if k != "_label"}
    if "." not in dim:
        raise ValueError(
            f"維度 {dim!r} 不是槽位路徑(缺少「.」),它的選項必須是 bundle"
            f"(key 含「.」的 dict),但得到:{option!r}"
        )
    return {dim: option}


def check_dimensions() -> None:
    """同一個槽位只能屬於一個維度,否則後套用的會安靜地蓋掉先套用的。"""
    owner: dict[str, str] = {}  # dotted_slot → 首見的維度名
    for dim, options in SLOT_OPTIONS.items():
        for opt in options:
            for slot in resolve_overrides(dim, opt):
                if owner.setdefault(slot, dim) != dim:
                    raise ValueError(
                        f"維度 {owner[slot]!r} 與 {dim!r} 都覆蓋槽位 {slot!r},"
                        "請把重疊的槽位合併進同一個維度(bundle)"
                    )


def dim_name(dim: str) -> str:
    """維度的顯示名:槽位路徑取槽位名,bundle 維度用原名。"""
    return dim.split(".")[1] if "." in dim else dim


def option_name(option: Any) -> str:
    """選項的簡短名稱(組 label 用)。"""
    if is_bundle(option):
        if option.get("_label"):
            return str(option["_label"])
        return "+".join(  # 沒給 _label 時的後備名:各槽位的選項名串起來
            f"{dim_name(slot)}:{option_name(opt)}"
            for slot, opt in option.items()
            if slot != "_label"
        )
    if isinstance(option, dict):
        return str(option.get("method", "custom"))
    if isinstance(option, list):
        return "+".join(map(str, option))
    return str(option)


def make_variants(base: dict) -> list[tuple[str, dict, dict]]:
    """回傳 [(label, overrides, config_dict), ...]。overrides 已展平為
    {dotted_slot: 選項}(bundle 展開、_label 移除),可直接重現該組合。"""
    check_dimensions()
    variants = []
    if MODE == "one_at_a_time":
        variants.append(("baseline", {}, copy.deepcopy(base)))
        for dim, options in SLOT_OPTIONS.items():
            for opt in options:
                cfg = copy.deepcopy(base)
                overrides = resolve_overrides(dim, opt)
                for dotted, o in overrides.items():
                    apply_option(cfg, dotted, o)
                variants.append((f"{dim_name(dim)}={option_name(opt)}", overrides, cfg))
    elif MODE == "product":
        dims = list(SLOT_OPTIONS)
        for combo in itertools.product(*(SLOT_OPTIONS[d] for d in dims)):
            cfg = copy.deepcopy(base)
            overrides: dict[str, Any] = {}
            for dim, opt in zip(dims, combo):
                overrides.update(resolve_overrides(dim, opt))
            for dotted, o in overrides.items():
                apply_option(cfg, dotted, o)
            label = " + ".join(
                f"{dim_name(d)}={option_name(o)}" for d, o in zip(dims, combo)
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

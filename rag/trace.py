"""逐步紀錄(trace):步驟順序與排版。

trace 的**收集**分別在 builder(ingestion / outer pipeline)與
MultiQueryRetrievalStage(inner pipeline);這裡負責另外兩件事:

1. :func:`step_order` —— 決定步驟的先後(依資料流,不是加入順序);
2. ``format_*`` —— 把 trace 排版成文字行。

排版之所以獨立成函式而不是直接 print:終端機與 log 檔要印同一份
內容(log 檔不截斷、終端機截斷),兩邊各寫一份必然會漂移。
"""

from __future__ import annotations

import logging
from typing import Any

import networkx

logger = logging.getLogger(__name__)


def step_order(pipeline: Any) -> list[str]:
    """回傳 pipeline 元件的執行順序(依資料流拓撲排序)。

    不能用 ``graph.nodes`` 的加入順序:builder 為了接線方便,元件的
    ``add_component`` 順序未必等於執行順序(例如 chunker 是條件式加入、
    排在 writer 之後),照加入順序印出來的紀錄會誤導。

    圖含環(Haystack 的 loop)而無法拓撲排序時,退回加入順序。
    """
    try:
        return list(networkx.topological_sort(pipeline.graph))
    except networkx.NetworkXUnfeasible:
        logger.warning(
            "fail-soft:pipeline 圖含環,無法拓撲排序;步驟順序退回元件"
            "加入順序(trace 的先後可能與實際執行不符)"
        )
        return list(pipeline.graph.nodes)


def snippet(text: str | None, width: int = 60) -> str:
    """壓縮空白後截斷,讓每筆切片佔一行。"""
    return " ".join((text or "").split())[:width]


def _doc_label(doc: Any) -> str:
    return str(doc.meta.get("chunk_id") or doc.id)


def _format_documents(
    documents: list[Any], limit: int, indent: str = "      "
) -> list[str]:
    """列出切片;``limit <= 0`` 表示全印(log 檔用)。"""
    shown = documents if limit <= 0 else documents[:limit]
    lines = [
        f"{indent}{rank:>2}. "
        f"score={'  n/a ' if doc.score is None else f'{doc.score:.4f}'} "
        f"[{_doc_label(doc)}] {snippet(doc.content)}"
        for rank, doc in enumerate(shown, start=1)
    ]
    if len(shown) < len(documents):
        lines.append(
            f"{indent}    …(其餘 {len(documents) - len(shown)} 筆略,"
            "--trace-docs 0 可全印)"
        )
    return lines


def format_ingestion_trace(trace: list[dict[str, Any]]) -> list[str]:
    """排版 ``run_ingestion()`` 的 trace:每步元件、實作類別與產出筆數。"""
    lines = ["--- 步驟 ---"]
    for step, entry in enumerate(trace, start=1):
        count = entry["count"]
        amount = "—" if count is None else f"{count} 筆"
        lines.append(
            f"  [{step}] {entry['component']:<10} {entry['type']:<28} → {amount}"
        )
        written = entry["output"].get("documents_written")
        if written is not None:
            lines.append(f"       documents_written={written}")
    return lines


def format_query_trace(trace: dict[str, Any], limit: int = 5) -> list[str]:
    """排版 ``query()`` 的 trace:改寫 → 各路檢索 → 每段重排 → 融合。

    Args:
        trace: ``query()`` 回傳的 ``trace`` 欄位。
        limit: 每步最多列出幾筆切片(``<= 0`` 為全印)。
    """
    lines = ["--- Query transformation ---", f"  [0] 原始查詢 → {trace['query']!r}"]
    if not trace["transforms"]:
        lines.append("      (無 transform,查詢原樣送入檢索)")
    for step, entry in enumerate(trace["transforms"], start=1):
        lines.append(f"  [{step}] {entry['component']} ({entry['type']})")
        lines.extend(f"      → {query!r}" for query in entry["queries"])

    # routing:獨立支線(吃原始查詢,不影響檢索)。用 .get:舊版 trace
    # dict(無 routing key)也要能排版。
    route = trace.get("routing")
    if route is not None:
        lines.append("")
        lines.append("--- Routing(查詢分類;不影響檢索)---")
        category = route.get("category")
        rest = {k: v for k, v in route.items() if k != "category"}
        detail = f" {rest}" if rest else ""
        lines.append(f"  → category={category!r}{detail}")

    subqueries = trace["subqueries"]
    for index, sub in enumerate(subqueries, start=1):
        lines.append("")
        lines.append(f"--- 子查詢 {index}/{len(subqueries)}:{sub['query']!r} ---")
        previous: list[Any] | None = None
        for step, entry in enumerate(sub["steps"], start=1):
            documents = entry["documents"]
            head = (
                f"  [{step}] {entry['component']} ({entry['type']})"
                f" → {len(documents)} 筆"
            )
            if previous is not None:
                kept = {_doc_label(doc) for doc in documents}
                dropped = [
                    _doc_label(doc) for doc in previous if _doc_label(doc) not in kept
                ]
                if dropped:
                    shown = ", ".join(dropped[:5])
                    more = "…" if len(dropped) > 5 else ""
                    head += f"(移除 {len(dropped)} 筆:{shown}{more})"
            lines.append(head)
            lines.extend(_format_documents(documents, limit))
            previous = documents

    fused = trace["fusion"]["documents"]
    applied = trace["fusion"]["applied"]
    lines.append("")
    if applied:
        lines.append("--- Fusion(去重 / 聚合 / 排序)---")
        lines.append(f"  → {len(fused)} 筆")
    elif applied is None:
        # custom fusion 一律執行,但 applied 只是建議輸出 —— 元件沒回報時
        # 講明「執行過、內容未知」,不猜測它做了什麼。
        lines.append("--- Fusion(custom 元件)---")
        lines.append(
            f"  → {len(fused)} 筆(custom fusion 一律執行;元件未回報 applied "
            "旗標 —— 建議在 @component.output_types 加上 applied: bool)"
        )
    else:
        # 元件恆存在於 pipeline,但單一查詢且 config 未設 fusion 時不做事;
        # 不講明的話,紀錄裡出現 fusion 會讓人以為結果被重排過。
        lines.append("--- Fusion(未啟用:原樣通過)---")
        lines.append(
            f"  → {len(fused)} 筆(單一查詢且 config 未設定 inference.fusion,"
            "未做去重 / 聚合 / 重新排序)"
        )
    lines.extend(_format_documents(fused, limit))

    # formatter:終端支線(對外格式)。用 .get:舊版 trace dict 也要能排版。
    formatted = trace.get("formatter")
    if formatted is not None:
        lines.append("")
        lines.append(f"--- Formatter(對外格式;{formatted['type']})---")
        # payload 是任意物件:終端機印截斷的 repr,log 檔(limit=0)全印
        rendered = repr(formatted["payload"])
        if limit > 0 and len(rendered) > 200:
            rendered = rendered[:200] + f"…(共 {len(rendered)} 字,log 檔全印)"
        lines.append(f"  → {rendered}")
    return lines

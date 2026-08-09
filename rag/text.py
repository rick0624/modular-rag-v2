"""LLM 回覆的 JSON 解析工具(fail-soft,供 LLM 元件共用)。"""

from __future__ import annotations

import json
import re
from typing import Any

_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """從 LLM 回覆取出第一個 JSON 物件。

    容忍 ```` ```json ```` code fence 與前後雜訊;解析失敗回傳
    ``None``,不 raise(供 fail-soft 的 LLM 元件共用)。
    """
    candidates = [text]
    fenced = _CODE_FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1))
    for candidate in candidates:
        stripped = candidate.strip()
        for attempt in (stripped, _brace_slice(stripped)):
            if not attempt:
                continue
            try:
                parsed = json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _brace_slice(text: str) -> str | None:
    """取首個 ``{`` 到末個 ``}`` 的切片(找不到成對大括號時回 None)。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]

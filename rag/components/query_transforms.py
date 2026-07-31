"""Query transformation 元件:一律 ``list[str] → list[str]``。

同一形狀讓方法鏈與 1→N 拆解天然相容:normalize 逐條處理、
llm_decompose 把一條拆成多條,串在一起就是「先正規化再拆解」。
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from haystack import component
from haystack.dataclasses import ChatMessage

from rag.errors import ComponentError

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


@component
class QueryNormalizer:
    """查詢正規化:NFKC(全形→半形)、壓縮空白、去頭尾、可選轉小寫。"""

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    @component.output_types(queries=list[str])
    def run(self, queries: list[str]) -> dict[str, Any]:
        normalized = []
        for text in queries:
            text = unicodedata.normalize("NFKC", text)
            text = _WHITESPACE.sub(" ", text).strip()
            if self.lowercase:
                text = text.lower()
            normalized.append(text)
        return {"queries": normalized}


@component
class GlossaryExpander:
    """術語表比對:查詢中出現的術語,其定義以 ``notes`` 輸出送進 prompt。

    ``expand_query=True`` 時,同時把「術語(定義)」附加到查詢文字,
    讓檢索也吃到展開後的內容(預設不改查詢,只補充 prompt)。
    這是客製擴充的骨架:換成公司的術語服務時,只需改 ``_load`` 與
    ``_match`` 的實作,輸入輸出形狀不變。
    """

    def __init__(
        self,
        glossary: dict[str, str] | None = None,
        glossary_path: str | None = None,
        expand_query: bool = False,
    ) -> None:
        self.expand_query = expand_query
        merged = dict(glossary or {})
        if glossary_path is not None:
            merged.update(self._load(glossary_path))
        if not merged:
            raise ComponentError(
                "glossary 方法需要術語表:請以 params.glossary 直接提供,"
                "或以 params.glossary_path 指定 YAML 檔(格式:術語: 定義)"
            )
        self.glossary = merged

    @staticmethod
    def _load(path: str) -> dict[str, str]:
        file = Path(path)
        if not file.is_file():
            raise ComponentError(f"找不到術語表檔案:{file}")
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ComponentError(
                f"術語表 '{file}' 必須是 YAML mapping(術語: 定義),"
                f"實際得到:{type(data).__name__}"
            )
        return {str(k): str(v) for k, v in data.items()}

    def _match(self, text: str) -> list[tuple[str, str]]:
        lowered = text.casefold()
        return [
            (term, definition)
            for term, definition in self.glossary.items()
            if term.casefold() in lowered
        ]

    @component.output_types(queries=list[str], notes=str)
    def run(self, queries: list[str]) -> dict[str, Any]:
        matched: dict[str, str] = {}
        expanded: list[str] = []
        for text in queries:
            hits = self._match(text)
            for term, definition in hits:
                matched.setdefault(term, definition)
            if self.expand_query and hits:
                additions = "、".join(f"{term}({definition})" for term, definition in hits)
                expanded.append(f"{text} {additions}")
            else:
                expanded.append(text)
        notes = "\n".join(f"{term}:{definition}" for term, definition in matched.items())
        return {"queries": expanded, "notes": notes}


DEFAULT_DECOMPOSE_PROMPT = """\
把下面的問題拆解成最多 {max_subqueries} 個可以獨立檢索的子問題。
規則:
- 每行輸出一個子問題,可加「1.」等編號。
- 問題本身已經夠簡單時,原樣輸出該問題即可。
- 除了子問題本身,不要輸出任何其他文字。

問題:{{ query }}"""


@component
class LLMQueryDecomposer:
    """LLM 查詢拆解:一條查詢 → N 條可獨立檢索的子查詢。

    LLM 掛掉或輸出無法解析時 **fail-soft**:記警告並退回原查詢
    (查詢路徑不可因 LLM 故障而中斷)。
    """

    def __init__(
        self,
        chat_generator: Any,
        prompt: str = DEFAULT_DECOMPOSE_PROMPT,
        max_subqueries: int = 4,
    ) -> None:
        self.chat_generator = chat_generator
        self.prompt = prompt
        self.max_subqueries = max_subqueries

    _NUMBERING = re.compile(r"^\s*(?:[-*•]|\d+[.)、]?)\s*")

    def _decompose(self, query: str) -> list[str]:
        prompt = self.prompt.replace("{{ query }}", query).replace(
            "{max_subqueries}", str(self.max_subqueries)
        )
        try:
            result = self.chat_generator.run(messages=[ChatMessage.from_user(prompt)])
            reply_text = result["replies"][0].text or ""
        except Exception as exc:  # fail-soft:LLM 故障不可中斷查詢路徑
            logger.warning("查詢拆解失敗(%s: %s),退回原查詢", type(exc).__name__, exc)
            return [query]
        subqueries = []
        for line in reply_text.splitlines():
            cleaned = self._NUMBERING.sub("", line).strip()
            if cleaned:
                subqueries.append(cleaned)
        if not subqueries:
            logger.warning("查詢拆解輸出無法解析(%r),退回原查詢", reply_text[:80])
            return [query]
        return subqueries[: self.max_subqueries]

    @component.output_types(queries=list[str])
    def run(self, queries: list[str]) -> dict[str, Any]:
        out: list[str] = []
        for query in queries:
            out.extend(self._decompose(query))
        return {"queries": out}

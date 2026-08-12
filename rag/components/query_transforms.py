"""Query transformation 元件:一律 ``list[str] → list[str]``。

同一形狀讓方法鏈與 1→N 拆解天然相容:normalize 逐條處理、
llm_decompose 把一條拆成多條,串在一起就是「先正規化再拆解」。
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from haystack import component
from haystack.dataclasses import ChatMessage

from rag.errors import ComponentError
from rag.text import extract_json_object

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


@component
class JargonMapper:
    """術語直白化:把查詢中出現的術語替換為對照表中的直白描述。

    與 :class:`GlossaryExpander` 的差異:glossary 保留原查詢、以
    ``notes`` 附註定義給 prompt;JargonMapper 直接**替換**查詢文字,
    適合檢索-only 或術語不利檢索的場景。
    """

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        json_path: str | None = None,
    ) -> None:
        merged = dict(mapping or {})
        if json_path is not None:
            merged.update(self._load(json_path))
        if not merged:
            raise ComponentError(
                "jargon_mapping 方法需要術語對照表:請以 params.mapping 直接提供,"
                "或以 params.json_path 指定 JSON 檔(格式:{\"術語\": \"直白描述\"})"
            )
        self.mapping = merged
        # 長 key 優先,避免「向量」搶走「向量檢索」的匹配;單趟 sub
        # 保證替換後的文字不會被其他 key 再次替換(無 cascading)。
        self._lookup = {term.casefold(): plain for term, plain in merged.items()}
        self._pattern = re.compile(
            "|".join(
                re.escape(term) for term in sorted(merged, key=len, reverse=True)
            ),
            re.IGNORECASE,
        )

    @staticmethod
    def _load(path: str) -> dict[str, str]:
        file = Path(path)
        if not file.is_file():
            raise ComponentError(f"找不到術語對照表檔案:{file}")
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ComponentError(
                f"術語對照表 '{file}' 不是合法的 JSON:{exc}"
            ) from exc
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in data.items()
        ):
            raise ComponentError(
                f"術語對照表 '{file}' 必須是扁平 JSON 物件"
                f"({{\"術語\": \"直白描述\"}}),實際得到:{type(data).__name__}"
            )
        return data

    @component.output_types(queries=list[str])
    def run(self, queries: list[str]) -> dict[str, Any]:
        replaced = [
            self._pattern.sub(
                lambda match: self._lookup[match.group(0).casefold()], text
            )
            for text in queries
        ]
        return {"queries": replaced}


DEFAULT_REWRITE_PROMPT = """\
你是文獻檢索專家。把下面的使用者問題改寫成一條乾淨、清晰、適合檢索的問題。
規則:
- 保留原意,不要新增問題中沒有的資訊。
- 不要改動專有名詞與術語。
- 只輸出 JSON,格式:{"rewritten_query": "改寫後的問題", "reason": "改寫理由"}。

問題:{{ query }}"""


@component
class LLMQueryRewriter:
    """LLM 查詢改寫:一條查詢 → 一條更適合檢索的查詢(1→1)。

    LLM 掛掉或輸出無法解析時 **fail-soft**:記警告並退回原查詢
    (查詢路徑不可因 LLM 故障而中斷)。
    """

    def __init__(
        self,
        chat_generator: Any,
        prompt: str = DEFAULT_REWRITE_PROMPT,
    ) -> None:
        self.chat_generator = chat_generator
        self.prompt = prompt

    def _rewrite(self, query: str) -> str:
        prompt = self.prompt.replace("{{ query }}", query)
        try:
            logger.debug("查詢改寫 prompt:\n%s", prompt)
            result = self.chat_generator.run(messages=[ChatMessage.from_user(prompt)])
            reply_text = result["replies"][0].text or ""
            logger.debug("查詢改寫回覆:%s", reply_text)
        except Exception as exc:  # fail-soft:LLM 故障不可中斷查詢路徑
            logger.warning("fail-soft:查詢改寫失敗(%s: %s),退回原查詢", type(exc).__name__, exc)
            return query
        parsed = extract_json_object(reply_text)
        rewritten = parsed.get("rewritten_query") if parsed else None
        if not isinstance(rewritten, str) or not rewritten.strip():
            logger.warning("fail-soft:查詢改寫輸出無法解析(%r),退回原查詢", reply_text[:80])
            return query
        return rewritten.strip()

    @component.output_types(queries=list[str])
    def run(self, queries: list[str]) -> dict[str, Any]:
        return {"queries": [self._rewrite(query) for query in queries]}


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
            logger.debug("查詢拆解 prompt:\n%s", prompt)
            result = self.chat_generator.run(messages=[ChatMessage.from_user(prompt)])
            reply_text = result["replies"][0].text or ""
            logger.debug("查詢拆解回覆:%s", reply_text)
        except Exception as exc:  # fail-soft:LLM 故障不可中斷查詢路徑
            logger.warning("fail-soft:查詢拆解失敗(%s: %s),退回原查詢", type(exc).__name__, exc)
            return [query]
        subqueries = []
        for line in reply_text.splitlines():
            cleaned = self._NUMBERING.sub("", line).strip()
            if cleaned:
                subqueries.append(cleaned)
        if not subqueries:
            logger.warning("fail-soft:查詢拆解輸出無法解析(%r),退回原查詢", reply_text[:80])
            return [query]
        return subqueries[: self.max_subqueries]

    @component.output_types(queries=list[str])
    def run(self, queries: list[str]) -> dict[str, Any]:
        out: list[str] = []
        for query in queries:
            out.extend(self._decompose(query))
        return {"queries": out}


DEFAULT_MULTI_HYDE_PROMPT = """\
你是知識庫寫手。請針對下面的問題,寫出 {num_documents} 段「角度互異」、
可能出現在參考文件中、且能回答該問題的短文(每段 100 字以內)。
規則:
- 各段落聚焦不同面向(如定義、步驟、比較、應用),彼此不重複。
- 內容力求具體;不確定的細節可以合理推測。
- 只輸出 JSON,格式:{"documents": ["短文一", "短文二"]}。

問題:{{ query }}"""


@component
class LLMMultiHyDEExpander:
    """Multi-HyDE:一條查詢 → k 篇角度互異的假設文件(FinNLP 2025)。

    HyDE 原理:假設文件與真實文件的語意對齊程度高於原始查詢,拿假設
    文件去做向量檢索命中率更高。Multi-HyDE 再加上「多角度」:k 篇假設
    文件各自檢索,結果由下游 fusion 合流。假設文件以字串流過 query
    通道,query embedder 對它做 embedding 即是 HyDE 的核心操作。
    LLM 掛掉或輸出無法解析時 **fail-soft**:記警告並退回原查詢。
    """

    def __init__(
        self,
        chat_generator: Any,
        prompt: str = DEFAULT_MULTI_HYDE_PROMPT,
        num_documents: int = 3,
        keep_original: bool = True,
    ) -> None:
        self.chat_generator = chat_generator
        self.prompt = prompt
        self.num_documents = num_documents
        self.keep_original = keep_original

    def _hypothesize(self, query: str) -> list[str]:
        """回傳假設文件(最多 num_documents 篇);失敗時回空 list。"""
        prompt = self.prompt.replace("{{ query }}", query).replace(
            "{num_documents}", str(self.num_documents)
        )
        try:
            logger.debug("Multi-HyDE prompt:\n%s", prompt)
            result = self.chat_generator.run(messages=[ChatMessage.from_user(prompt)])
            reply_text = result["replies"][0].text or ""
            logger.debug("Multi-HyDE 回覆:%s", reply_text)
        except Exception as exc:  # fail-soft:LLM 故障不可中斷查詢路徑
            logger.warning(
                "fail-soft:Multi-HyDE 生成失敗(%s: %s),退回原查詢", type(exc).__name__, exc
            )
            return []
        parsed = extract_json_object(reply_text)
        documents = parsed.get("documents") if parsed else None
        if not isinstance(documents, list):
            logger.warning(
                "fail-soft:Multi-HyDE 輸出無法解析(%r),退回原查詢", reply_text[:80]
            )
            return []
        valid = [doc.strip() for doc in documents if isinstance(doc, str) and doc.strip()]
        if not valid:
            logger.warning("fail-soft:Multi-HyDE 沒有可用的假設文件,退回原查詢")
        return valid[: self.num_documents]

    @component.output_types(queries=list[str])
    def run(self, queries: list[str]) -> dict[str, Any]:
        out: list[str] = []
        for query in queries:
            hypothetical = self._hypothesize(query)
            if self.keep_original or not hypothetical:
                out.append(query)
            out.extend(hypothetical)
        return {"queries": out}


DEFAULT_PREQRAG_CLASSIFY_PROMPT = """\
判斷下面的問題屬於哪一類:
- single:問的是單一事實或主題,一份文件就能回答。
- multi:需要比較、彙整或跨多份文件的資訊才能回答。
規則:
- 只輸出 JSON,格式:{"type": "single"} 或 {"type": "multi"}。

問題:{{ query }}"""

DEFAULT_PREQRAG_REWRITE_PROMPT = """\
你是檢索專家。把下面的問題改寫成 {num_rewrites} 條說法不同、意思相同、
更適合檢索的問題。
規則:
- 保留原意,不要新增問題中沒有的資訊,不要改動專有名詞。
- 只輸出 JSON,格式:{"rewrites": ["改寫一", "改寫二"]}。

問題:{{ query }}"""


@component
class PreQRAGDispatcher:
    """PreQRAG:先分類、再分派(arXiv 2506.17493, SIGIR 2025 LiveRAG)。

    每條查詢三步:(1) LLM 分類 single / multi;(2) single → 改寫成
    ``num_rewrites`` 條檢索取向的問題;(3) multi → 以
    :class:`LLMQueryDecomposer` 拆解成子查詢。動機:簡單問題被拆解反而
    稀釋檢索、複雜問題只改寫又不夠力,分類後各走適合的路。
    fail-soft:分類失敗視為 single、改寫失敗退回原查詢
    (拆解的 fail-soft 由 LLMQueryDecomposer 保證)。
    """

    def __init__(
        self,
        chat_generator: Any,
        classify_prompt: str = DEFAULT_PREQRAG_CLASSIFY_PROMPT,
        rewrite_prompt: str = DEFAULT_PREQRAG_REWRITE_PROMPT,
        decompose_prompt: str = DEFAULT_DECOMPOSE_PROMPT,
        num_rewrites: int = 2,
        max_subqueries: int = 4,
        include_original: bool = True,
    ) -> None:
        self.chat_generator = chat_generator
        self.classify_prompt = classify_prompt
        self.rewrite_prompt = rewrite_prompt
        self.num_rewrites = num_rewrites
        self.include_original = include_original
        self._decomposer = LLMQueryDecomposer(
            chat_generator=chat_generator,
            prompt=decompose_prompt,
            max_subqueries=max_subqueries,
        )

    def _ask(self, prompt: str) -> str | None:
        """送出 prompt 並回傳回覆文字;LLM 掛掉時回 None(fail-soft)。"""
        try:
            logger.debug("PreQRAG prompt:\n%s", prompt)
            result = self.chat_generator.run(messages=[ChatMessage.from_user(prompt)])
            reply_text = result["replies"][0].text or ""
            logger.debug("PreQRAG 回覆:%s", reply_text)
            return reply_text
        except Exception as exc:  # fail-soft:LLM 故障不可中斷查詢路徑
            logger.warning("fail-soft:PreQRAG LLM 呼叫失敗(%s: %s)", type(exc).__name__, exc)
            return None

    def _classify(self, query: str) -> str:
        reply = self._ask(self.classify_prompt.replace("{{ query }}", query))
        parsed = extract_json_object(reply) if reply is not None else None
        qtype = parsed.get("type") if parsed else None
        if qtype not in ("single", "multi"):
            logger.warning(
                "fail-soft:PreQRAG 分類無法解析(%r),視為 single", (reply or "")[:80]
            )
            return "single"
        return qtype

    def _rewrite(self, query: str) -> list[str]:
        prompt = self.rewrite_prompt.replace("{{ query }}", query).replace(
            "{num_rewrites}", str(self.num_rewrites)
        )
        reply = self._ask(prompt)
        parsed = extract_json_object(reply) if reply is not None else None
        rewrites = parsed.get("rewrites") if parsed else None
        if not isinstance(rewrites, list):
            logger.warning(
                "fail-soft:PreQRAG 改寫無法解析(%r),退回原查詢", (reply or "")[:80]
            )
            return [query]
        valid = [r.strip() for r in rewrites if isinstance(r, str) and r.strip()]
        return valid[: self.num_rewrites] or [query]

    @component.output_types(queries=list[str])
    def run(self, queries: list[str]) -> dict[str, Any]:
        out: list[str] = []
        for query in queries:
            if self._classify(query) == "multi":
                expanded = self._decomposer.run(queries=[query])["queries"]
            else:
                expanded = self._rewrite(query)
            expanded = [text for text in expanded if text != query]  # 與原查詢重複的不重複檢索
            if self.include_original:
                out.append(query)
                out.extend(expanded)
            else:
                out.extend(expanded or [query])  # 查詢不可變空
        return {"queries": out}

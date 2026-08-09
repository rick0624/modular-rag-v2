"""LLM 事實查核元件:過濾與查詢無關的檢索切片。

屬於 reranking 槽位的方法(可串接在 similarity / llm 之後),遵守
reranking 契約:只過濾,不重排、不改分、不修改內容。
"""

from __future__ import annotations

import logging
from typing import Any

from haystack import Document, component
from haystack.dataclasses import ChatMessage

from rag.text import extract_json_object

logger = logging.getLogger(__name__)

DEFAULT_FACT_CHECK_PROMPT = """\
判斷下列每個片段是否與問題相關(內容可用於回答問題才算相關)。
規則:
- 只輸出 JSON,格式:{"relevant_indices": [1, 3]},列出相關片段的 1 起編號。
- 沒有任何相關片段時輸出 {"relevant_indices": []}。

問題:{{ query }}

片段:
{{ documents }}"""


@component
class LLMFactChecker:
    """LLM 事實查核:listwise 判斷相關性,只過濾、不重排、不改分。

    一次呼叫把 query 與全部切片送給 LLM,依回覆的 ``relevant_indices``
    移除不相關者;存活切片保持原輸入順序與原分數。LLM 掛掉或輸出
    無法解析時 **fail-soft**:記警告並保留全部切片。
    """

    def __init__(
        self,
        chat_generator: Any,
        prompt: str = DEFAULT_FACT_CHECK_PROMPT,
        max_docs: int | None = None,
    ) -> None:
        self.chat_generator = chat_generator
        self.prompt = prompt
        self.max_docs = max_docs

    def _relevant_indices(self, query: str, documents: list[Document]) -> list[int] | None:
        """回傳 0 起的存活索引;無法判斷時回 None(保留全部)。"""
        rendered = "\n".join(
            f"{i}. {doc.content or ''}" for i, doc in enumerate(documents, start=1)
        )
        prompt = self.prompt.replace("{{ query }}", query).replace(
            "{{ documents }}", rendered
        )
        try:
            logger.debug("事實查核 prompt:\n%s", prompt)
            result = self.chat_generator.run(messages=[ChatMessage.from_user(prompt)])
            reply_text = result["replies"][0].text or ""
            logger.debug("事實查核回覆:%s", reply_text)
        except Exception as exc:  # fail-soft:LLM 故障不可中斷查詢路徑
            logger.warning(
                "fail-soft:事實查核失敗(%s: %s),保留全部切片", type(exc).__name__, exc
            )
            return None
        parsed = extract_json_object(reply_text)
        indices = parsed.get("relevant_indices") if parsed else None
        if not isinstance(indices, list):
            logger.warning(
                "fail-soft:事實查核輸出無法解析(%r),保留全部切片", reply_text[:80]
            )
            return None
        survivors: list[int] = []
        for item in indices:
            if isinstance(item, int) and not isinstance(item, bool) and 1 <= item <= len(documents):
                survivors.append(item - 1)
            else:
                logger.warning("事實查核回覆含無效編號 %r(有效範圍 1-%d),已忽略", item, len(documents))
        return sorted(set(survivors))

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document]) -> dict[str, Any]:
        if not documents:
            return {"documents": []}
        checked = documents if self.max_docs is None else documents[: self.max_docs]
        passthrough = [] if self.max_docs is None else documents[self.max_docs :]
        survivors = self._relevant_indices(query, checked)
        if survivors is None:
            return {"documents": list(documents)}
        if not survivors:
            logger.info("事實查核判定所有受檢切片皆與查詢無關")
        kept = [checked[i] for i in survivors]
        dropped = [
            doc.meta.get("chunk_id") or doc.id
            for i, doc in enumerate(checked)
            if i not in set(survivors)
        ]
        if dropped:
            logger.info(
                "事實查核移除 %d/%d 筆(判定不相關):%s",
                len(dropped), len(checked), ", ".join(str(item) for item in dropped),
            )
        return {"documents": kept + passthrough}

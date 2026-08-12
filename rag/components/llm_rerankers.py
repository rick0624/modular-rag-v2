"""InsertRank 重排元件:附檢索分數的 listwise LLM 重排。

屬於 reranking 槽位的方法,training-free(純 prompt,不需微調模型)。
"""

from __future__ import annotations

import logging
from typing import Any

from haystack import Document, component
from haystack.dataclasses import ChatMessage

from rag.text import extract_json_object

logger = logging.getLogger(__name__)

DEFAULT_INSERTRANK_PROMPT = """\
你是檢索結果重排專家。請依「與問題的相關程度」由高到低,重新排列下列片段。
每個片段開頭的 [{score_label}] 是檢索系統給它的分數,可作為詞彙相關性的
參考訊號;請以內容判斷為主、分數為輔。
規則:
- 只輸出 JSON,格式:{"ranking": [2, 1, 3]},列出**全部**片段的 1 起編號,
  最相關的在最前面。
- 不要輸出任何其他文字。

問題:{{ query }}

片段:
{{ documents }}"""


@component
class InsertRankLLMRanker:
    """InsertRank:附檢索分數的 listwise 重排(arXiv 2506.14086, WSDM 2026)。

    與一般 listwise LLM 重排的差異:prompt 中每個候選片段都附上檢索器
    給的分數(BM25 等),讓 LLM 同時對「內容語意」與「詞彙訊號」推理。
    分數的身分取決於上游:retrieval 為 ``bm25`` 時最忠實於論文;hybrid
    經 RRF 融合後分數量級不同,可用 ``score_label`` 據實描述。
    只重排、不改分、不改內容;LLM 掛掉或輸出無法解析時 **fail-soft**:
    記警告並保留原順序(仍收斂到 top_k)。
    """

    def __init__(
        self,
        chat_generator: Any,
        prompt: str = DEFAULT_INSERTRANK_PROMPT,
        top_k: int = 5,
        score_label: str = "檢索分數",
    ) -> None:
        self.chat_generator = chat_generator
        self.prompt = prompt
        self.top_k = top_k
        self.score_label = score_label

    def _render(self, documents: list[Document]) -> str:
        lines = []
        for i, doc in enumerate(documents, start=1):
            score = "無" if doc.score is None else f"{doc.score:.2f}"
            lines.append(f"{i}. [{self.score_label}: {score}] {doc.content or ''}")
        return "\n".join(lines)

    def _ranking(self, query: str, documents: list[Document]) -> list[int] | None:
        """回傳 0 起的名次索引;無法判斷時回 None(保留原順序)。"""
        prompt = (
            self.prompt.replace("{{ query }}", query)
            .replace("{{ documents }}", self._render(documents))
            .replace("{score_label}", self.score_label)
        )
        try:
            logger.debug("InsertRank prompt:\n%s", prompt)
            result = self.chat_generator.run(messages=[ChatMessage.from_user(prompt)])
            reply_text = result["replies"][0].text or ""
            logger.debug("InsertRank 回覆:%s", reply_text)
        except Exception as exc:  # fail-soft:LLM 故障不可中斷查詢路徑
            logger.warning(
                "fail-soft:InsertRank 重排失敗(%s: %s),保留原順序", type(exc).__name__, exc
            )
            return None
        parsed = extract_json_object(reply_text)
        ranking = parsed.get("ranking") if parsed else None
        if not isinstance(ranking, list):
            logger.warning(
                "fail-soft:InsertRank 輸出無法解析(%r),保留原順序", reply_text[:80]
            )
            return None
        ordered: list[int] = []
        for item in ranking:
            if (
                isinstance(item, int)
                and not isinstance(item, bool)
                and 1 <= item <= len(documents)
                and item - 1 not in ordered
            ):
                ordered.append(item - 1)
            else:
                logger.warning(
                    "InsertRank 回覆含無效或重複編號 %r(有效範圍 1-%d),已忽略",
                    item, len(documents),
                )
        return ordered

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document]) -> dict[str, Any]:
        if not documents:
            return {"documents": []}
        ordered = self._ranking(query, documents)
        if ordered is None:
            return {"documents": list(documents)[: self.top_k]}
        # 未被列出的片段視為較不相關,依原順序補在後面
        remainder = [i for i in range(len(documents)) if i not in set(ordered)]
        reranked = [documents[i] for i in ordered + remainder]
        return {"documents": reranked[: self.top_k]}

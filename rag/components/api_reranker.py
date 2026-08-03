"""Reranking 方法:api_rerank(通用 HTTP rerank API,請求 / 回應形狀可設定)。

公司內部或第三方的 rerank API 欄位名稱各不相同,本元件以參數對應,
不必為每種服務寫新程式:

| API 形狀 | 設定 |
|---|---|
| 請求 ``{"question", "documents", "model"}`` / 回應 ``{"returnData": [{"index", "score"}]}`` | 預設值即可 |
| 請求 ``{"query", "documents"}`` | ``query_field: query`` |
| 回應 ``{"results": [{"index", "relevance_score"}]}``(Cohere 式) | ``results_field: results`` + ``score_field: relevance_score`` |
| 回應 ``[{"index", "score"}]``(回應本身就是清單) | ``results_field: null`` |

契約(與 reranking 槽位其他方法一致):

- 輸入 ``query`` + ``documents``,輸出重排後的 ``documents``;
  ``index`` 是文件在**送出清單**中的位置,回應未列出的文件視同淘汰。
- 分數寫回 ``Document.score`` —— 下游 fusion 的 ``max_score`` 會用到,
  但 API 分數的量綱未知,跨 subquery 比較請用預設的 ``rrf``。
- **fail-soft**:API 掛掉或回應無法解析時記警告並保留原檢索順序,
  查詢路徑不中斷(初次接線想讓錯誤直接炸出來:``raise_on_failure: true``)。
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import httpx

from haystack import Document, component

from rag.components.http_json import locate_list, post_json
from rag.errors import APIResponseFormatError, RAGFrameworkError

logger = logging.getLogger(__name__)


@component
class FlexibleAPIRanker:
    """通用 HTTP rerank API:一次把 query 與全部候選送出,依回傳分數重排。"""

    def __init__(
        self,
        *,
        endpoint: str,
        headers: dict[str, str] | None = None,
        model: str | None = None,
        top_k: int = 5,
        timeout: float = 30.0,
        query_field: str = "question",
        documents_field: str = "documents",
        model_field: str = "model",
        results_field: str | None = "returnData",
        index_field: str = "index",
        score_field: str = "score",
        index_base: int = 0,
        higher_is_better: bool = True,
        raise_on_failure: bool = False,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.headers = dict(headers or {})
        self.model = model
        self.top_k = top_k
        self.timeout = timeout
        self.query_field = query_field
        self.documents_field = documents_field
        self.model_field = model_field
        self.results_field = results_field
        self.index_field = index_field
        self.score_field = score_field
        self.index_base = index_base
        self.higher_is_better = higher_is_better
        self.raise_on_failure = raise_on_failure
        self._client = client  # 可注入以便測試;None 時每次請求各自建立

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document]) -> dict[str, Any]:
        if not documents:
            return {"documents": []}
        try:
            ranked = self._rerank(query, documents)
        except RAGFrameworkError as exc:
            if self.raise_on_failure:
                raise
            # fail-soft:重排失敗仍有檢索順序可用,不讓查詢整條掛掉。
            # 這行走 WARNING,終端機預設就看得到 —— 靜默降級最難查。
            logger.warning(
                "rerank API 失敗(%s: %s),保留原檢索順序前 %d 筆",
                type(exc).__name__, exc, self.top_k,
            )
            return {"documents": list(documents[: self.top_k])}
        return {"documents": ranked}

    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        body: dict[str, Any] = {
            self.query_field: query,
            self.documents_field: [doc.content or "" for doc in documents],
        }
        if self.model is not None:
            body[self.model_field] = self.model
        logger.debug("rerank API 請求:%d 筆候選,query=%r", len(documents), query)
        data = post_json(
            self.endpoint,
            body,
            headers=self.headers,
            timeout=self.timeout,
            client=self._client,
        )
        results = locate_list(
            data,
            self.results_field,
            api_label="rerank API ",
            setting_name="results_field",
            what="重排結果清單",
        )
        scored = self._parse_results(results, len(documents))
        # API 不保證已排序,一律自己排;兩段穩定排序 = 同分時回到送出順序
        # (只排分數的話,同分順序會取決於 API 回應的排列,不可重現)。
        scored.sort(key=lambda pair: pair[0])
        scored.sort(key=lambda pair: pair[1], reverse=self.higher_is_better)
        if len(scored) < len(documents):
            logger.info(
                "rerank API 回傳 %d/%d 筆,未列出的視同淘汰",
                len(scored), len(documents),
            )
        return [
            dataclasses.replace(documents[position], score=score)
            for position, score in scored[: self.top_k]
        ]

    def _parse_results(self, results: list[Any], total: int) -> list[tuple[int, float]]:
        """把回應清單轉成 ``(送出清單中的位置, 分數)``,並擋掉無效項目。

        Raises:
            APIResponseFormatError: 元素不是物件、缺欄位、index 型別不對,
                或 index 全數越界(多半是 ``index_base`` 設反了)。
        """
        pairs: list[tuple[int, float]] = []
        seen: set[int] = set()
        out_of_range: list[int] = []
        for order, item in enumerate(results):
            if not isinstance(item, dict):
                raise APIResponseFormatError(
                    f"rerank API 結果清單的第 {order} 個元素必須是物件,"
                    f"實際得到:{type(item).__name__}"
                )
            for field in (self.index_field, self.score_field):
                if field not in item:
                    raise APIResponseFormatError(
                        f"rerank API 結果的第 {order} 個元素缺少 '{field}' 欄位;"
                        f"實際的欄位:{sorted(item.keys())}。"
                        "請用 index_field / score_field 對應你的 API 回應"
                    )
            raw_index = item[self.index_field]
            if not isinstance(raw_index, int) or isinstance(raw_index, bool):
                raise APIResponseFormatError(
                    f"rerank API 結果的第 {order} 個元素的 "
                    f"'{self.index_field}' 必須是整數,實際得到:{raw_index!r}"
                )
            try:
                score = float(item[self.score_field])
            except (TypeError, ValueError) as exc:
                raise APIResponseFormatError(
                    f"rerank API 結果的第 {order} 個元素的 "
                    f"'{self.score_field}' 不是數字:{item[self.score_field]!r}"
                ) from exc
            position = raw_index - self.index_base
            if not 0 <= position < total:
                out_of_range.append(raw_index)
                continue
            if position in seen:  # 重複的名次:保留先出現者
                logger.warning("rerank API 回傳重複的 index %d,已忽略後者", raw_index)
                continue
            seen.add(position)
            pairs.append((position, score))
        if out_of_range and not pairs:
            raise APIResponseFormatError(
                f"rerank API 回傳的 index 全數越界(送出 {total} 筆,收到 "
                f"{out_of_range[:5]});index_base 目前是 {self.index_base},"
                f"若你的 API 是 {1 - self.index_base} 起算請改設 index_base: "
                f"{1 - self.index_base}"
            )
        if out_of_range:
            logger.warning(
                "rerank API 回傳 %d 個越界的 index(送出 %d 筆:%s),已忽略",
                len(out_of_range), total, out_of_range[:5],
            )
        return pairs

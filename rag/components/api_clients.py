"""公司 / 第三方 HTTP API 的客戶端元件:embedding 與 rerank(形狀可設定)。

外部 API 的請求 / 回應欄位各不相同,這裡以參數對應,不必為每種服務寫
新程式(對映表見 README「api_embedding / api_rerank 形狀對映表」節):

- :func:`post_json` / :func:`locate_list`:共同的 HTTP/JSON 請求路徑,
  錯誤一律翻成框架例外,訊息帶 endpoint 與回應內容前 200 字。
- :class:`FlexibleAPITextEmbedder` / :class:`FlexibleAPIDocumentEmbedder`
  (``embedding: api_embedding``):回應形狀以 embeddings_field /
  item_field 對應。
- :class:`FlexibleAPIRanker`(``reranking: api_rerank``):請求 / 回應
  形狀以 query_field / results_field / index_base 等對應;**fail-soft**
  —— API 掛掉時記警告並保留原檢索順序,查詢不中斷。
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import httpx

from haystack import Document, component

from rag.errors import APICallError, APIResponseFormatError, RAGFrameworkError

logger = logging.getLogger(__name__)


def post_json(
    endpoint: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> Any:
    """POST 一個 JSON 請求並回傳解析後的回應。

    Args:
        endpoint: 完整的請求 URL。
        body: 請求主體(由 httpx 序列化,Content-Type 自動帶上)。
        headers: 額外的 HTTP 標頭(認證放這裡)。
        timeout: 逾時秒數。
        client: 可注入的 httpx client(供測試);None 時每次請求各自建立。

    Raises:
        APICallError: timeout、連線失敗或非 2xx 狀態碼。
        APIResponseFormatError: 回應不是合法的 JSON。
    """
    try:
        if client is not None:
            response = client.post(
                endpoint, json=body, headers=headers, timeout=timeout
            )
        else:
            response = httpx.post(
                endpoint, json=body, headers=headers, timeout=timeout
            )
    except httpx.TimeoutException as exc:
        raise APICallError(f"呼叫 API 逾時({timeout} 秒):POST {endpoint}") from exc
    except httpx.HTTPError as exc:
        raise APICallError(f"呼叫 API 失敗:POST {endpoint}({exc})") from exc
    if not 200 <= response.status_code < 300:
        preview = response.text[:200]
        raise APICallError(
            f"API 回應非 2xx 狀態碼 {response.status_code}:POST {endpoint}"
            f"(回應內容前 200 字:{preview!r})"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise APIResponseFormatError(
            f"API 回應不是合法的 JSON:POST {endpoint}"
        ) from exc


def locate_list(
    data: Any,
    path: str | None,
    *,
    api_label: str,
    setting_name: str,
    what: str,
) -> list[Any]:
    """依 ``a.b`` 路徑從回應中取出一個清單。

    找不到時,錯誤訊息會列出該層實際存在的欄位 —— 對照公司 API 的
    回應調設定,不必猜。

    Args:
        path: 巢狀路徑;``None`` 表示回應本身就是清單。
        api_label: 訊息中的 API 名稱(例如 "embedding API")。
        setting_name: 訊息中要調整的設定欄位名。
        what: 訊息中該清單的內容描述(例如 "向量清單")。

    Raises:
        APIResponseFormatError: 路徑不存在,或該處的值不是 list。
    """
    current: Any = data
    if path is not None:
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                hint = (
                    f"該層實際的欄位:{sorted(current.keys())}"
                    if isinstance(current, dict)
                    else f"該層實際型別:{type(current).__name__}"
                )
                raise APIResponseFormatError(
                    f"{api_label}回應中找不到 '{path}'(在 '{part}' 處中斷);{hint}。"
                    f"請把 {setting_name} 設成你的 API 回應中{what}所在的欄位"
                    "(支援 a.b 巢狀路徑;回應本身就是清單時設為 null)"
                )
            current = current[part]
    if not isinstance(current, list):
        location = (
            f"'{path}' 的值"
            if path is not None
            else f"{setting_name} 為 null 時,回應本身"
        )
        raise APIResponseFormatError(
            f"{location}必須是 list,實際得到:{type(current).__name__}"
        )
    return current

class _FlexibleAPIEmbedderCore:
    """Text / Document 兩個元件共用的請求與回應解析邏輯。"""

    def __init__(
        self,
        *,
        endpoint: str,
        headers: dict[str, str] | None = None,
        model: str | None = None,
        batch_size: int = 16,
        timeout: float = 30.0,
        texts_field: str = "input",
        model_field: str = "model",
        embeddings_field: str | None = "embeddings",
        item_field: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.headers = dict(headers or {})
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.texts_field = texts_field
        self.model_field = model_field
        self.embeddings_field = embeddings_field
        self.item_field = item_field
        self._client = client  # 可注入以便測試;None 時每批請求各自建立

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """分批呼叫 API 並串接回傳的向量。

        Raises:
            APICallError: timeout、連線失敗或非 2xx 狀態碼。
            APIResponseFormatError: 回應缺少向量欄位或長度不符。
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            body: dict[str, Any] = {self.texts_field: batch}
            if self.model is not None:
                body[self.model_field] = self.model
            data = self._request_json(body)
            batch_vectors = self._locate_embeddings(data)
            if len(batch_vectors) != len(batch):
                raise APIResponseFormatError(
                    f"embedding API 回傳的向量數({len(batch_vectors)})"
                    f"與輸入文字數({len(batch)})不符"
                )
            vectors.extend(
                self._to_vector(item, position)
                for position, item in enumerate(batch_vectors)
            )
        return vectors

    def _request_json(self, body: dict[str, Any]) -> Any:
        return post_json(
            self.endpoint,
            body,
            headers=self.headers,
            timeout=self.timeout,
            client=self._client,
        )

    def _locate_embeddings(self, data: Any) -> list[Any]:
        """依 ``embeddings_field`` 路徑從回應中取出向量清單。"""
        return locate_list(
            data,
            self.embeddings_field,
            api_label="embedding API ",
            setting_name="embeddings_field",
            what="向量清單",
        )

    def _to_vector(self, item: Any, position: int) -> list[float]:
        """把向量清單的單一元素轉成 ``list[float]``。

        元素可以是數字 list,或(設定 ``item_field`` 時)含向量欄位的物件。
        """
        value: Any = item
        if isinstance(item, dict):
            if self.item_field is None:
                raise APIResponseFormatError(
                    f"向量清單的第 {position} 個元素是物件"
                    f"(欄位:{sorted(item.keys())}),請設定 item_field 指出"
                    "向量所在的欄位(例如 OpenAI 式回應設為 embedding)"
                )
            if self.item_field not in item:
                raise APIResponseFormatError(
                    f"向量清單的第 {position} 個元素缺少 "
                    f"'{self.item_field}' 欄位;實際的欄位:{sorted(item.keys())}"
                )
            value = item[self.item_field]
        if not isinstance(value, list):
            raise APIResponseFormatError(
                f"第 {position} 個向量必須是數字 list,實際得到:{type(value).__name__}"
            )
        try:
            return [float(element) for element in value]
        except (TypeError, ValueError) as exc:
            raise APIResponseFormatError(f"第 {position} 個向量含有非數字元素") from exc


@component
class FlexibleAPITextEmbedder:
    """查詢端 API embedder:單一查詢文字 → 向量。"""

    def __init__(self, *, client: httpx.Client | None = None, **kwargs: Any) -> None:
        self._core = _FlexibleAPIEmbedderCore(client=client, **kwargs)

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, Any]:
        return {"embedding": self._core.embed_texts([text])[0]}


@component
class FlexibleAPIDocumentEmbedder:
    """文件端 API embedder:為每個 Document 填入向量(依 batch_size 分批)。"""

    def __init__(self, *, client: httpx.Client | None = None, **kwargs: Any) -> None:
        self._core = _FlexibleAPIEmbedderCore(client=client, **kwargs)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        vectors = self._core.embed_texts([doc.content or "" for doc in documents])
        return {
            "documents": [
                dataclasses.replace(doc, embedding=vector)
                for doc, vector in zip(documents, vectors)
            ]
        }

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
                "fail-soft:rerank API 失敗(%s: %s),保留原檢索順序前 %d 筆",
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

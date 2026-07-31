"""Embedding 方法:api_embedding(通用 HTTP API,回應形狀可設定)。

公司內部或第三方的 embedding API 回應結構各不相同;本元件以三個欄位
設定對應,不必為每種服務寫新程式:

| API 回應結構 | 設定 |
|---|---|
| ``{"embeddings": [[...], ...]}`` | 預設值即可 |
| ``{"result": {"embeddings": [[...], ...]}}`` | ``embeddings_field: result.embeddings`` |
| ``{"data": [{"embedding": [...]}, ...]}``(OpenAI 式) | ``embeddings_field: data`` + ``item_field: embedding`` |
| ``[[...], ...]``(回應本身就是清單,如 HuggingFace TEI) | ``embeddings_field: null`` |

欄位對不上時,錯誤訊息會列出回應中實際存在的欄位,照著調整設定即可。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import httpx

from haystack import Document, component

from rag.errors import APICallError, APIResponseFormatError


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
        try:
            if self._client is not None:
                response = self._client.post(
                    self.endpoint, json=body, headers=self.headers, timeout=self.timeout
                )
            else:
                response = httpx.post(
                    self.endpoint, json=body, headers=self.headers, timeout=self.timeout
                )
        except httpx.TimeoutException as exc:
            raise APICallError(
                f"呼叫 API 逾時({self.timeout} 秒):POST {self.endpoint}"
            ) from exc
        except httpx.HTTPError as exc:
            raise APICallError(
                f"呼叫 API 失敗:POST {self.endpoint}({exc})"
            ) from exc
        if not 200 <= response.status_code < 300:
            preview = response.text[:200]
            raise APICallError(
                f"API 回應非 2xx 狀態碼 {response.status_code}:POST {self.endpoint}"
                f"(回應內容前 200 字:{preview!r})"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise APIResponseFormatError(
                f"API 回應不是合法的 JSON:POST {self.endpoint}"
            ) from exc

    def _locate_embeddings(self, data: Any) -> list[Any]:
        """依 ``embeddings_field`` 路徑從回應中取出向量清單。

        找不到時,錯誤訊息會列出該層實際存在的欄位,方便對照 API 回應
        結構調整設定。
        """
        current: Any = data
        if self.embeddings_field is not None:
            for part in self.embeddings_field.split("."):
                if not isinstance(current, dict) or part not in current:
                    hint = (
                        f"該層實際的欄位:{sorted(current.keys())}"
                        if isinstance(current, dict)
                        else f"該層實際型別:{type(current).__name__}"
                    )
                    raise APIResponseFormatError(
                        f"embedding API 回應中找不到 '{self.embeddings_field}'"
                        f"(在 '{part}' 處中斷);{hint}。"
                        "請把 embeddings_field 設成你的 API 回應中向量清單所在的欄位"
                        "(支援 a.b 巢狀路徑;回應本身就是清單時設為 null)"
                    )
                current = current[part]
        if not isinstance(current, list):
            location = (
                f"'{self.embeddings_field}' 的值"
                if self.embeddings_field is not None
                else "embeddings_field 為 null 時,回應本身"
            )
            raise APIResponseFormatError(
                f"{location}必須是 list,實際得到:{type(current).__name__}"
            )
        return current

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

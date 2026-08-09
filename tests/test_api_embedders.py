"""FlexibleAPI embedder 測試:請求體組裝、回應形狀對映、診斷錯誤訊息。"""

from __future__ import annotations

import httpx
import pytest
from conftest import FakeClient, FakeResponse

from haystack import Document

from rag.components.api_clients import (
    FlexibleAPIDocumentEmbedder,
    FlexibleAPITextEmbedder,
)
from rag.errors import APICallError, APIResponseFormatError

ENDPOINT = "https://api.example.com/v1/embeddings"


def _text_embedder(client, **kwargs):
    return FlexibleAPITextEmbedder(client=client, endpoint=ENDPOINT, **kwargs)


class TestRequestBody:
    def test_texts_field_and_model(self):
        client = FakeClient([FakeResponse(payload={"embeddings": [[1.0, 2.0]]})])
        embedder = _text_embedder(client, model="my-model", texts_field="texts")
        result = embedder.run(text="哈囉")
        assert result["embedding"] == [1.0, 2.0]
        assert client.calls[0]["json"] == {"texts": ["哈囉"], "model": "my-model"}

    def test_model_omitted_when_none(self):
        client = FakeClient([FakeResponse(payload={"embeddings": [[0.5]]})])
        _text_embedder(client).run(text="q")
        assert "model" not in client.calls[0]["json"]

    def test_headers_forwarded(self):
        client = FakeClient([FakeResponse(payload={"embeddings": [[0.5]]})])
        _text_embedder(client, headers={"Authorization": "Bearer T"}).run(text="q")
        assert client.calls[0]["headers"]["Authorization"] == "Bearer T"


class TestResponseShapes:
    def test_nested_dot_path(self):
        client = FakeClient(
            [FakeResponse(payload={"result": {"embeddings": [[1.0]]}})]
        )
        embedder = _text_embedder(client, embeddings_field="result.embeddings")
        assert embedder.run(text="q")["embedding"] == [1.0]

    def test_openai_style_item_field(self):
        client = FakeClient(
            [FakeResponse(payload={"data": [{"embedding": [3.0, 4.0]}]})]
        )
        embedder = _text_embedder(client, embeddings_field="data", item_field="embedding")
        assert embedder.run(text="q")["embedding"] == [3.0, 4.0]

    def test_bare_list_response(self):
        client = FakeClient([FakeResponse(payload=[[9.0]])])
        embedder = _text_embedder(client, embeddings_field=None)
        assert embedder.run(text="q")["embedding"] == [9.0]


class TestDiagnostics:
    def test_missing_field_lists_actual_keys(self):
        client = FakeClient(
            [FakeResponse(payload={"data": [], "model": "m", "usage": {}})]
        )
        embedder = _text_embedder(client)  # 預設 embeddings_field="embeddings"
        with pytest.raises(
            APIResponseFormatError, match=r"該層實際的欄位:\['data', 'model', 'usage'\]"
        ):
            embedder.run(text="q")

    def test_dict_item_without_item_field_names_the_knob(self):
        client = FakeClient(
            [FakeResponse(payload={"embeddings": [{"embedding": [1.0]}]})]
        )
        with pytest.raises(APIResponseFormatError, match="請設定 item_field"):
            _text_embedder(client).run(text="q")

    def test_length_mismatch(self):
        client = FakeClient([FakeResponse(payload={"embeddings": [[1.0], [2.0]]})])
        with pytest.raises(APIResponseFormatError, match="向量數(.*)與輸入文字數"):
            _text_embedder(client).run(text="q")

    def test_non_2xx_includes_status_and_preview(self):
        client = FakeClient([FakeResponse(status_code=500, text="internal boom")])
        with pytest.raises(APICallError, match="500(.*)internal boom"):
            _text_embedder(client).run(text="q")

    def test_timeout_wrapped(self):
        client = FakeClient([httpx.ConnectTimeout("slow")])
        with pytest.raises(APICallError, match="逾時"):
            _text_embedder(client).run(text="q")


class TestDocumentEmbedder:
    def test_batches_and_preserves_ids(self):
        client = FakeClient(
            [
                FakeResponse(payload={"embeddings": [[1.0], [2.0]]}),
                FakeResponse(payload={"embeddings": [[3.0]]}),
            ]
        )
        embedder = FlexibleAPIDocumentEmbedder(
            client=client, endpoint=ENDPOINT, batch_size=2
        )
        docs = [
            Document(id=f"d{i}::chunk_0", content=f"內容 {i}", meta={"doc_id": f"d{i}"})
            for i in range(3)
        ]
        out = embedder.run(documents=docs)["documents"]
        assert [d.embedding for d in out] == [[1.0], [2.0], [3.0]]
        assert [d.id for d in out] == ["d0::chunk_0", "d1::chunk_0", "d2::chunk_0"]
        assert len(client.calls) == 2, "batch_size=2 時 3 份文件應分兩批"

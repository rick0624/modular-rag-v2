"""Embedding 方法:mock(離線測試用,確定性偽向量,不需下載模型)。

向量由 token hash 疊加產生:同一段文字永遠得到同一個向量,且詞彙
重疊的文字向量相似,足以驗證檢索流程。維度由 ``dim`` 控制。

Haystack 2.32 的 core 也會內建同名 mock embedder;本框架保留自己的
版本以維持 v1 語意(hash 疊加、CJK bigram),import 路徑不衝突。
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any

from haystack import Document, component

from rag.text import tokenize


def mock_vector(text: str, dim: int) -> list[float]:
    """為一段文字產生確定性向量(每個 token 依 hash 投影後疊加、L2 正規化)。"""
    vector = [0.0] * dim
    tokens = tokenize(text) or [""]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        vector[0] = 1.0
        norm = 1.0
    return [value / norm for value in vector]


@component
class MockTextEmbedder:
    """查詢端 mock embedder:文字 → 確定性向量。"""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, Any]:
        return {"embedding": mock_vector(text, self.dim)}


@component
class MockDocumentEmbedder:
    """文件端 mock embedder:為每個 Document 填入確定性向量。"""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        return {
            "documents": [
                dataclasses.replace(
                    doc, embedding=mock_vector(doc.content or "", self.dim)
                )
                for doc in documents
            ]
        }

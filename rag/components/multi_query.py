"""多子查詢檢索:內部單查詢 pipeline 的迴圈包裝。

Haystack 的圖是靜態的,無法對「執行期才知道數量」的子查詢開 N 條分支;
標準解法是把「單一查詢的檢索(+重排)」組成一條內部 pipeline,由本元件
對每條子查詢各跑一次 —— 保留 v1 語意:**各子查詢獨立檢索與重排,
之後才由 fusion 合併**。內部 pipeline 仍是原生 Haystack 物件,
可單獨測試與視覺化。
"""

from __future__ import annotations

from typing import Any

from haystack import Document, Pipeline, component


@component
class MultiQueryRetrievalStage:
    """對每條子查詢執行內部檢索 pipeline,輸出各路(已重排)結果。

    Args:
        inner: 單查詢 pipeline(輸入為查詢文字,輸出為 documents)。
        query_targets: 查詢文字要餵給內部 pipeline 的哪些 socket,
            例如 ``[("query_embedder", "text"), ("ranker", "query")]``。
        output_component: 內部 pipeline 中產出最終 documents 的元件名稱。
    """

    def __init__(
        self,
        inner: Pipeline,
        query_targets: list[tuple[str, str]],
        output_component: str,
    ) -> None:
        self.inner = inner
        self.query_targets = list(query_targets)
        self.output_component = output_component

    @component.output_types(results=list[list[Document]])
    def run(self, queries: list[str]) -> dict[str, Any]:
        results: list[list[Document]] = []
        for query in queries:
            data = {
                component_name: {input_name: query}
                for component_name, input_name in self.query_targets
            }
            output = self.inner.run(data)
            results.append(output[self.output_component]["documents"])
        return {"results": results}

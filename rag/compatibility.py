"""pipeline 建構期的方法組合相容性檢查(語意層)。

Haystack 的 socket 型別驗證只保證「接得起來」,不保證「組合有意義」
(例如 PDF importer 的輸出只有 PDF converter 能處理)。相容性以
方法型錄 factory 表上的宣告式欄位表達:

- import 方法宣告 ``output_content_type_fn``;parsing 方法宣告
  ``input_content_types`` 與 ``produces_pages``。
- chunking 方法宣告 ``requires_pages``。
- indexing 方法宣告 ``capabilities``;retrieval 方法宣告
  ``required_capabilities``。
- custom 方法的宣告寫在 config 參數裡,由呼叫端讀出後以值傳入。

不相容時丟出 :class:`~rag.errors.IncompatiblePipelineError`,並掃描
factory 表列出目前可相容的方法。新增相容性維度時,在
:class:`~rag.slots.SlotFactory` 加宣告欄位、在這裡補一條檢查即可。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rag.errors import IncompatiblePipelineError


def _format_set(values: frozenset[str] | set[str]) -> str:
    """把集合排序後格式化,讓錯誤訊息穩定可讀。"""
    return "{" + ", ".join(repr(v) for v in sorted(values)) + "}"


def _list_names(names: list[str]) -> str:
    return ", ".join(repr(name) for name in sorted(names)) or "(目前沒有相容的方法)"


def validate_ingestion_compatibility(
    import_method: str,
    output_type: str | None,
    parsing_method: str,
    parser_input_types: frozenset[str],
    parsing_table: Mapping[str, Any],
) -> None:
    """檢查 import 方法輸出的 content_type 是否能被 parsing 鏈的第一個方法處理。

    ``output_type`` 與 ``parser_input_types`` 由呼叫端提供:內建方法讀
    factory 的宣告,custom 方法讀 config 參數的宣告(builder 的
    ``parsing_declaration``)。

    Raises:
        IncompatiblePipelineError: parser 不支援 importer 的輸出型別;
            訊息列出可處理該型別的 parsing 方法。
    """
    if output_type is None or output_type in parser_input_types:
        return
    compatible = [
        name
        for name, factory in parsing_table.items()
        if factory.kind == "converter" and output_type in factory.input_content_types
    ]
    raise IncompatiblePipelineError(
        f"import 方法 '{import_method}' 輸出 content_type {_format_set({output_type})},"
        f"但 parsing 方法 '{parsing_method}' 只支援 "
        f"{_format_set(parser_input_types)}"
        f"(無法處理 {_format_set({output_type})})。"
        f"可相容的 parsing 方法:{_list_names(compatible)}"
    )


def validate_chunking_compatibility(
    parsing_methods: list[str],
    chain_produces_pages: bool,
    chunking_method: str,
    chunker_requires_pages: bool,
    parsing_table: Mapping[str, Any],
    chunking_table: Mapping[str, Any],
) -> None:
    """檢查 chunking 方法需要的分頁輸入是否由 parsing 鏈提供。

    方法鏈的 ``produces_pages`` 為「任一環節產生分頁即成立」;
    ``chunker_requires_pages`` 由呼叫端提供(custom 方法的宣告在 config
    參數裡,內建方法在 factory 上)。

    Raises:
        IncompatiblePipelineError: chunker 需要分頁但 parsing 鏈不產生;
            訊息列出會產生分頁的 parsing 方法與不需分頁的 chunking 方法。
    """
    if not chunker_requires_pages or chain_produces_pages:
        return
    paginated_parsers = [
        name for name, factory in parsing_table.items() if factory.produces_pages
    ]
    non_paginated_chunkers = [
        name for name, factory in chunking_table.items() if not factory.requires_pages
    ]
    chain_display = (
        parsing_methods[0] if len(parsing_methods) == 1 else str(parsing_methods)
    )
    raise IncompatiblePipelineError(
        f"chunking 方法 '{chunking_method}' 需要分頁輸入(pages),但 parsing "
        f"方法 '{chain_display}' 不產生分頁資訊。"
        f"會產生分頁的 parsing 方法:{_list_names(paginated_parsers)};"
        f"或改用不需分頁的 chunking 方法:{_list_names(non_paginated_chunkers)}"
    )


def validate_inference_compatibility(
    retrieval_method: str,
    retriever: Any,
    indexing_method: str,
    indexer: Any,
    retrieval_table: Mapping[str, Any],
) -> None:
    """檢查 retrieval 方法需要的索引能力是否都由 indexing 方法提供。

    Raises:
        IncompatiblePipelineError: indexer 缺少 retriever 需要的能力;
            訊息列出此索引可支援的 retrieval 方法。
    """
    missing = retriever.required_capabilities - indexer.capabilities
    if not missing:
        return
    compatible = [
        name
        for name, factory in retrieval_table.items()
        if factory.required_capabilities <= indexer.capabilities
    ]
    raise IncompatiblePipelineError(
        f"retrieval 方法 '{retrieval_method}' 需要索引能力 "
        f"{_format_set(retriever.required_capabilities)},但 indexing 方法 "
        f"'{indexing_method}' 只提供 {_format_set(indexer.capabilities)}"
        f"(缺少 {_format_set(missing)})。"
        f"此索引可相容的 retrieval 方法:{_list_names(compatible)}"
    )

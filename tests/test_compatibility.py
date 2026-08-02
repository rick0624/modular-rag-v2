"""相容性檢查測試:三個維度都要在建構期擋下,並列出可用替代。"""

from __future__ import annotations

import pytest
from conftest import make_config

from rag.builder import SlotFactory, build_ingestion_pipeline
from rag.compatibility import validate_inference_compatibility
from rag.config import parse_config
from rag.errors import IncompatiblePipelineError


def test_pdf_import_with_plain_text_parser_rejected():
    config = parse_config(
        make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": "./x", "extensions": [".pdf"]},
                },
                "parsing": {"method": "plain_text"},
            }
        )
    )
    with pytest.raises(IncompatiblePipelineError) as excinfo:
        build_ingestion_pipeline(config)
    message = str(excinfo.value)
    assert "import 方法 'local_file' 輸出 content_type {'pdf'}" in message
    assert "只支援 {'text'}" in message
    assert "'auto'" in message and "'pdf'" in message  # 相容選項含分流與純 pdf


def test_page_based_requires_paginating_parser():
    config = parse_config(
        make_config(ingestion={"chunking": {"method": "page_based"}})
    )
    with pytest.raises(IncompatiblePipelineError) as excinfo:
        build_ingestion_pipeline(config)
    message = str(excinfo.value)
    assert "chunking 方法 'page_based' 需要分頁輸入" in message
    assert "會產生分頁的 parsing 方法:'auto', 'pdf'" in message
    assert "'fixed_size'" in message  # 列出不需分頁的替代 chunking 方法


def test_paginating_chain_link_satisfies_page_requirement(pdf_dir):
    # 鏈中任一環節產生分頁即可(pdf converter + clean 處理器)
    config = parse_config(
        make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(pdf_dir), "extensions": [".pdf"]},
                },
                "parsing": {"method": ["pdf", "clean"], "method_params": {}},
                "chunking": {"method": "page_based"},
            }
        )
    )
    build_ingestion_pipeline(config)  # 不應拋錯


def test_index_capability_check_lists_alternatives():
    weak_index = SlotFactory(build=lambda p, c: None, capabilities=frozenset({"text_search"}))
    hybrid_retriever = SlotFactory(
        build=lambda p, c: None,
        required_capabilities=frozenset({"vector_search", "text_search"}),
    )
    table = {
        "bm25": SlotFactory(
            build=lambda p, c: None,
            required_capabilities=frozenset({"text_search"}),
        ),
        "hybrid": hybrid_retriever,
    }
    with pytest.raises(IncompatiblePipelineError) as excinfo:
        validate_inference_compatibility(
            "hybrid", hybrid_retriever, "weak_index", weak_index, table
        )
    message = str(excinfo.value)
    assert "需要索引能力 {'text_search', 'vector_search'}" in message
    assert "缺少 {'vector_search'}" in message
    assert "此索引可相容的 retrieval 方法:'bm25'" in message

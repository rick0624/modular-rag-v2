"""custom module 機制:載入、槽位契約驗證、鏈中組合、端到端。"""

from __future__ import annotations

import pytest
from conftest import make_config

from rag.builder import build_pipelines
from rag.config import parse_config
from rag.errors import ConfigError, UnknownMethodError

# ---------------------------------------------------------------------------
# 測試用元件原始碼(寫進 tmp_path 的 .py 檔)
# ---------------------------------------------------------------------------

TRANSFORM_OK = '''
from typing import Any
from haystack import component


@component
class SuffixTransform:
    """合約正確的 transform:每條查詢加上後綴。"""

    def __init__(self, suffix: str = "!") -> None:
        self.suffix = suffix

    @component.output_types(queries=list[str])
    def run(self, queries: list[str]) -> dict[str, Any]:
        return {"queries": [q + self.suffix for q in queries]}


class _PrivateHelper:
    pass
'''

RETRIEVER_OK = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class FakeCompanyRetriever:
    """模擬公司檢索:回傳帶 dockey meta 的 Document。"""

    @component.output_types(documents=list[Document])
    def run(self, query: str) -> dict[str, Any]:
        return {"documents": [
            Document(
                content=f"關於「{query}」的內部文件內容",
                score=0.9,
                meta={"dockey": "DOC-1", "doc_id": "DOC-1",
                      "chunk_id": "DOC-1::chunk_0"},
            )
        ]}
'''

RERANKER_OK = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class ReverseReranker:
    """合約正確的 reranker:單純反轉順序(可觀察生效)。"""

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document]) -> dict[str, Any]:
        return {"documents": list(reversed(documents))}
'''

NOT_A_COMPONENT = '''
class PlainClass:
    def run(self, queries):
        return {"queries": queries}
'''

MISSING_INPUT = '''
from typing import Any
from haystack import component


@component
class NoQueriesInput:
    @component.output_types(queries=list[str])
    def run(self, texts: list[str]) -> dict[str, Any]:
        return {"queries": texts}
'''

MISSING_OUTPUT = '''
from typing import Any
from haystack import component


@component
class WrongOutputName:
    @component.output_types(results=list[str])
    def run(self, queries: list[str]) -> dict[str, Any]:
        return {"results": queries}
'''

WRONG_INPUT_TYPE = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class WrongType:
    @component.output_types(queries=list[str])
    def run(self, queries: list[Document]) -> dict[str, Any]:
        return {"queries": []}
'''

EXTRA_MANDATORY_INPUT = '''
from typing import Any
from haystack import component


@component
class NeedsMode:
    @component.output_types(queries=list[str])
    def run(self, queries: list[str], mode: str) -> dict[str, Any]:
        return {"queries": queries}
'''

EXTRA_OUTPUT_ALLOWED = '''
from typing import Any
from haystack import component


@component
class WithDebugOutput:
    @component.output_types(queries=list[str], debug=dict[str, Any])
    def run(self, queries: list[str]) -> dict[str, Any]:
        return {"queries": queries, "debug": {"count": len(queries)}}
'''

BROKEN_IMPORT = '''
import nonexistent_module_xyz_for_test
'''


def write_module(tmp_path, source: str, name: str = "custom_mod.py") -> str:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


def custom_transform_config(corpus_dir, file: str, cls: str, init_params=None):
    params = {"file": file, "class": cls}
    if init_params is not None:
        params["init_params"] = init_params
    return parse_config(
        make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {
                        "input_dir": str(corpus_dir),
                        "extensions": [".txt"],
                    },
                }
            },
            inference={
                "query_transformation": {"method": "custom", "params": params}
            },
        )
    )


# ---------------------------------------------------------------------------
# 載入與端到端
# ---------------------------------------------------------------------------


class TestLoading:
    def test_custom_transform_file_end_to_end(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, TRANSFORM_OK)
        config = custom_transform_config(
            corpus_dir, file, "SuffixTransform", {"suffix": "?"}
        )
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()
        result = pipelines.query("向量檢索")
        assert result["trace"]["transforms"][0]["queries"] == ["向量檢索?"]
        assert result["answer"] is not None

    def test_custom_via_class_path(self):
        # dotted path 載入既有元件(QueryNormalizer 天然滿足 transform 契約)
        config = parse_config(
            make_config(
                inference={
                    "query_transformation": {
                        "method": "custom",
                        "params": {
                            "class_path": (
                                "rag.components.query_transforms:QueryNormalizer"
                            ),
                            "init_params": {"lowercase": True},
                        },
                    }
                }
            )
        )
        pipelines = build_pipelines(config)
        component = pipelines.inference.get_component("transform")
        assert type(component).__name__ == "QueryNormalizer"

    def test_custom_retrieval_end_to_end(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, RETRIEVER_OK)
        config = parse_config(
            make_config(
                ingestion={
                    "import": {
                        "method": "local_file",
                        "params": {
                            "input_dir": str(corpus_dir),
                            "extensions": [".txt"],
                        },
                    }
                },
                inference={
                    "retrieval": {
                        "method": "custom",
                        "params": {"file": file, "class": "FakeCompanyRetriever"},
                    },
                    "reranking": {"method": "none"},
                },
            )
        )
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()
        result = pipelines.query("請假規則")
        assert result["documents"]
        assert result["documents"][0].meta["dockey"] == "DOC-1"

    def test_custom_in_reranking_chain(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, RERANKER_OK)
        config = parse_config(
            make_config(
                ingestion={
                    "import": {
                        "method": "local_file",
                        "params": {
                            "input_dir": str(corpus_dir),
                            "extensions": [".txt"],
                        },
                    }
                },
                inference={
                    "reranking": {
                        "method": ["custom", "llm"],
                        "method_params": {
                            "custom": {"file": file, "class": "ReverseReranker"},
                            "llm": {"top_k": 3},
                        },
                    }
                },
            )
        )
        pipelines = build_pipelines(config)
        inner = pipelines.inference.get_component("multi_query").inner
        assert type(inner.get_component("ranker")).__name__ == "ReverseReranker"
        assert type(inner.get_component("ranker_2")).__name__ == "LLMRanker"
        pipelines.run_ingestion()
        result = pipelines.query("向量檢索")
        assert result["documents"]

    def test_extra_output_socket_allowed(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, EXTRA_OUTPUT_ALLOWED)
        config = custom_transform_config(corpus_dir, file, "WithDebugOutput")
        pipelines = build_pipelines(config)  # 建構期不報錯即通過
        pipelines.run_ingestion()
        result = pipelines.query("查詢")
        assert result["documents"] is not None

    def test_unknown_method_still_lists_custom(self):
        config = parse_config(
            make_config(
                inference={"query_transformation": {"method": "sustom"}}
            )
        )
        with pytest.raises(UnknownMethodError, match="'custom'"):
            build_pipelines(config)


# ---------------------------------------------------------------------------
# 載入失敗的錯誤訊息
# ---------------------------------------------------------------------------


def build_with_transform_params(params: dict) -> None:
    config = parse_config(
        make_config(
            inference={
                "query_transformation": {"method": "custom", "params": params}
            }
        )
    )
    build_pipelines(config)


class TestLoadErrors:
    def test_file_not_found_message(self, tmp_path):
        missing = str(tmp_path / "no_such.py")
        with pytest.raises(ConfigError, match="找不到檔案"):
            build_with_transform_params({"file": missing, "class": "X"})

    def test_class_not_in_file_lists_available(self, tmp_path):
        file = write_module(tmp_path, TRANSFORM_OK)
        with pytest.raises(ConfigError, match="'SuffixTransform'") as excinfo:
            build_with_transform_params({"file": file, "class": "Nope"})
        # 列出檔內實際定義的公開類別;私有類別不列
        assert "_PrivateHelper" not in str(excinfo.value)

    def test_import_error_wrapped(self, tmp_path):
        file = write_module(tmp_path, BROKEN_IMPORT)
        with pytest.raises(ConfigError, match="nonexistent_module_xyz_for_test"):
            build_with_transform_params({"file": file, "class": "X"})

    def test_class_path_without_colon_rejected(self):
        with pytest.raises(ConfigError, match="pkg.mod:ClassName"):
            build_with_transform_params(
                {"class_path": "rag.components.query_transforms.QueryNormalizer"}
            )

    def test_class_path_module_not_importable(self):
        with pytest.raises(ConfigError, match="無法 import 模組"):
            build_with_transform_params(
                {"class_path": "no_such_package_xyz:Whatever"}
            )

    def test_params_require_exactly_one_source(self, tmp_path):
        file = write_module(tmp_path, TRANSFORM_OK)
        # 都不給
        with pytest.raises(ConfigError, match="恰好指定一個"):
            build_with_transform_params({})
        # 都給
        with pytest.raises(ConfigError, match="恰好指定一個"):
            build_with_transform_params(
                {"class_path": "a:B", "file": file, "class": "SuffixTransform"}
            )
        # file 缺 class
        with pytest.raises(ConfigError, match="class 指定"):
            build_with_transform_params({"file": file})

    def test_init_params_mismatch_shows_signature(self, tmp_path):
        file = write_module(tmp_path, TRANSFORM_OK)
        with pytest.raises(ConfigError, match="建構子簽名") as excinfo:
            build_with_transform_params(
                {
                    "file": file,
                    "class": "SuffixTransform",
                    "init_params": {"bogus": 1},
                }
            )
        assert "suffix" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 槽位契約驗證
# ---------------------------------------------------------------------------


class TestContractValidation:
    def test_not_a_component_rejected(self, tmp_path):
        file = write_module(tmp_path, NOT_A_COMPONENT)
        with pytest.raises(ConfigError, match="@component"):
            build_with_transform_params({"file": file, "class": "PlainClass"})

    def test_missing_input_socket_message(self, tmp_path):
        file = write_module(tmp_path, MISSING_INPUT)
        with pytest.raises(ConfigError, match="缺少輸入 socket") as excinfo:
            build_with_transform_params({"file": file, "class": "NoQueriesInput"})
        message = str(excinfo.value)
        assert "queries" in message  # 缺什麼
        assert "texts" in message  # 實際有什麼

    def test_missing_output_socket_message(self, tmp_path):
        file = write_module(tmp_path, MISSING_OUTPUT)
        with pytest.raises(ConfigError, match="缺少輸出 socket") as excinfo:
            build_with_transform_params({"file": file, "class": "WrongOutputName"})
        message = str(excinfo.value)
        assert "queries" in message
        assert "output_types" in message

    def test_wrong_socket_type_message(self, tmp_path):
        file = write_module(tmp_path, WRONG_INPUT_TYPE)
        with pytest.raises(ConfigError, match="型別") as excinfo:
            build_with_transform_params({"file": file, "class": "WrongType"})
        assert "list[str]" in str(excinfo.value)

    def test_extra_mandatory_input_rejected(self, tmp_path):
        file = write_module(tmp_path, EXTRA_MANDATORY_INPUT)
        with pytest.raises(ConfigError, match="契約以外的必填輸入") as excinfo:
            build_with_transform_params({"file": file, "class": "NeedsMode"})
        message = str(excinfo.value)
        assert "'mode'" in message
        assert "init_params" in message  # 指出修法

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

GENERATOR_OK = '''
from typing import Any
from haystack import component
from haystack.dataclasses import ChatMessage


@component
class EchoGenerator:
    """合約正確的 chat generator:回覆帶上收到的 messages 內容。"""

    def __init__(self, prefix: str = "[custom]") -> None:
        self.prefix = prefix

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        joined = " | ".join(m.text or "" for m in messages)
        return {"replies": [ChatMessage.from_assistant(
            f"{self.prefix} {joined}", meta={"model": "echo", "custom": True}
        )]}
'''

GENERATOR_WRONG_SHAPE = '''
from typing import Any
from haystack import component


@component
class NotAChatGenerator:
    """把 generation 槽位誤當成 transform 寫(常見誤解)。"""

    @component.output_types(queries=list[str])
    def run(self, queries: list[str]) -> dict[str, Any]:
        return {"queries": queries}
'''

FUSION_OK = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class TagOnlyFusion:
    """合約正確的 fusion(刻意不輸出 applied):攤平 + 打標記。"""

    @component.output_types(documents=list[Document])
    def run(self, results: list[list[Document]]) -> dict[str, Any]:
        merged = [d for docs in results for d in docs]
        for doc in merged:
            doc.meta["fused_by"] = "TagOnlyFusion"
        return {"documents": merged}
'''

FUSION_WRONG_SHAPE = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class FlatInputFusion:
    """把 fusion 誤當 reranker 寫(吃單層 documents)。"""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        return {"documents": documents}
'''

IMPORTER_BYTESTREAM_OK = '''
from pathlib import Path
from typing import Any
from haystack import component
from haystack.dataclasses import ByteStream


@component
class FakeApiImporter:
    """合約正確的 importer:ByteStream 內容 + 帶 doc_id 的 meta。"""

    @component.output_types(
        sources=list[str | Path | ByteStream], meta=list[dict[str, Any]]
    )
    def run(self) -> dict[str, Any]:
        rows = [("API-1", "請假需於三日前提出申請。"), ("API-2", "薪資每月五日發放。")]
        return {
            "sources": [ByteStream(data=text.encode("utf-8")) for _, text in rows],
            "meta": [{"doc_id": key} for key, _ in rows],
        }
'''

IMPORTER_NEEDS_INPUT = '''
from pathlib import Path
from typing import Any
from haystack import component
from haystack.dataclasses import ByteStream


@component
class NeedsQueryImporter:
    @component.output_types(
        sources=list[str | Path | ByteStream], meta=list[dict[str, Any]]
    )
    def run(self, root: str) -> dict[str, Any]:
        return {"sources": [], "meta": []}
'''

IMPORTER_NO_DOC_ID = '''
from pathlib import Path
from typing import Any
from haystack import component
from haystack.dataclasses import ByteStream


@component
class NoDocIdImporter:
    @component.output_types(
        sources=list[str | Path | ByteStream], meta=list[dict[str, Any]]
    )
    def run(self) -> dict[str, Any]:
        return {
            "sources": [ByteStream(data="內容".encode("utf-8"))],
            "meta": [{"title": "沒帶 doc_id"}],
        }
'''

CONVERTER_OK = '''
from pathlib import Path
from typing import Any
from haystack import component
from haystack.dataclasses import ByteStream, Document


@component
class EnvelopeParser:
    """合約正確的鏈首 converter:sources + meta → documents。"""

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: list[dict[str, Any]],
    ) -> dict[str, Any]:
        documents = []
        for source, entry in zip(sources, meta):
            text = (
                source.data.decode("utf-8")
                if isinstance(source, ByteStream)
                else Path(source).read_text(encoding="utf-8")
            )
            documents.append(Document(content=text, meta=dict(entry)))
        return {"documents": documents}
'''

CONVERTER_NARROW_SOURCES = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class PathOnlyParser:
    """sources 型別註記過窄(list[str]):pipeline 可能送 ByteStream。"""

    @component.output_types(documents=list[Document])
    def run(self, sources: list[str], meta: list[dict[str, Any]]) -> dict[str, Any]:
        return {"documents": []}
'''

DOC_PROCESSOR_OK = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class UpperCaser:
    """合約正確的鏈中文件處理器:documents → documents。"""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        return {"documents": [
            Document(content=(d.content or "").upper(), meta=dict(d.meta))
            for d in documents
        ]}
'''

CHUNKER_OK = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class LineChunker:
    """合約正確的 chunker:一行一塊,meta 逐塊複製(doc_id 保留)。"""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        chunks = []
        for doc in documents:
            for line in (doc.content or "").splitlines():
                if line.strip():
                    chunks.append(Document(content=line, meta=dict(doc.meta)))
        return {"documents": chunks}
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


class TestCustomGeneration:
    """generation 槽位:契約是 Haystack ChatGenerator 形狀(messages → replies)。"""

    @staticmethod
    def _config(corpus_dir, file: str, **extra_params):
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
                    "reranking": {"method": "none"},
                    "generation": {
                        "method": "custom",
                        "params": {
                            "file": file,
                            "class": "EchoGenerator",
                            **extra_params,
                        },
                    },
                },
            )
        )

    def test_custom_generation_end_to_end(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, GENERATOR_OK)
        config = self._config(
            corpus_dir, file, init_params={"prefix": "[公司 LLM]"}
        )
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()

        result = pipelines.query("向量檢索")
        assert result["answer"].startswith("[公司 LLM]")
        # meta 原樣往外帶(事後查得到答案是誰生的)
        assert result["reply_meta"]["custom"] is True
        assert result["trace"]["generation"]["answer"] == result["answer"]

    def test_prompt_options_still_apply(self, tmp_path, corpus_dir):
        """prompt 由框架組:system_prompt / prompt_template 與內建方法同款。"""
        file = write_module(tmp_path, GENERATOR_OK)
        config = self._config(
            corpus_dir,
            file,
            system_prompt="你是測試助理。",
            # 自訂模板一樣要用到 documents(prompt_builder 的 socket 由
            # 模板變數決定),與內建方法的限制相同
            prompt_template="模板:{{ query }}/{{ documents|length }} 筆",
        )
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()

        result = pipelines.query("向量檢索")
        assert "你是測試助理。" in result["prompt"]
        assert "模板:向量檢索/" in result["prompt"]
        # 元件真的收到組好的兩則 messages(system + user)
        assert "你是測試助理。 | 模板:向量檢索/" in result["answer"]

    def test_custom_generator_reused_by_llm_transform(self, tmp_path, corpus_dir):
        """params.generator 也吃 custom:整條 pipeline 只走公司的推論服務。"""
        file = write_module(tmp_path, GENERATOR_OK)
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
                    "query_transformation": {
                        "method": "llm_rewrite",
                        "params": {
                            "generator": {
                                "method": "custom",
                                "params": {"file": file, "class": "EchoGenerator"},
                            }
                        },
                    },
                    "reranking": {"method": "none"},
                },
            )
        )
        pipelines = build_pipelines(config)
        rewriter = pipelines.inference.get_component("transform")
        assert type(rewriter.chat_generator).__name__ == "EchoGenerator"

    def test_wrong_shape_message_points_at_messages_socket(
        self, tmp_path, corpus_dir
    ):
        file = write_module(tmp_path, GENERATOR_WRONG_SHAPE)
        config = parse_config(
            make_config(
                inference={
                    "generation": {
                        "method": "custom",
                        "params": {"file": file, "class": "NotAChatGenerator"},
                    }
                }
            )
        )
        with pytest.raises(ConfigError) as excinfo:
            build_pipelines(config)
        message = str(excinfo.value)
        assert "缺少輸入 socket 'messages" in message
        assert "queries" in message  # 實際有什麼


class TestCustomFusion:
    """fusion 槽位:契約 results: list[list[Document]] → documents。"""

    @staticmethod
    def _config(corpus_dir, fusion):
        data = make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                }
            },
            inference={"reranking": {"method": "none"}},
        )
        data["inference"]["fusion"] = fusion
        return parse_config(data)

    def test_custom_fusion_end_to_end_single_query(self, tmp_path, corpus_dir):
        """掛了 custom 就一律執行:單一查詢(N=1)也進元件。"""
        file = write_module(tmp_path, FUSION_OK)
        config = self._config(
            corpus_dir,
            {"method": "custom", "params": {"file": file, "class": "TagOnlyFusion"}},
        )
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()

        result = pipelines.query("向量檢索")
        assert result["documents"], "fusion 應把單路結果原樣攤平"
        assert all(
            doc.meta["fused_by"] == "TagOnlyFusion" for doc in result["documents"]
        ), "N=1 也應執行 custom fusion(不做內建穿透)"
        # 元件未輸出 applied → trace 三態的 None(不是 False)
        assert result["trace"]["fusion"]["applied"] is None

    def test_wrong_shape_message_points_at_results(self, tmp_path):
        file = write_module(tmp_path, FUSION_WRONG_SHAPE)
        data = make_config()
        data["inference"]["fusion"] = {
            "method": "custom", "params": {"file": file, "class": "FlatInputFusion"}
        }
        with pytest.raises(ConfigError) as excinfo:
            build_pipelines(parse_config(data))
        assert "缺少輸入 socket 'results" in str(excinfo.value)

    def test_flat_config_still_builds_builtin(self, corpus_dir):
        config = self._config(corpus_dir, {"strategy": "rrf", "top_k": 2})
        pipelines = build_pipelines(config)
        fusion = pipelines.inference.get_component("fusion")
        assert type(fusion).__name__ == "SubqueryFusion"
        assert fusion.always_fuse is True  # 有 fusion 區塊 = 單查詢也融合


FORMATTER_OK = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class EnvelopeFormatter:
    """合約正確的 formatter:公司信封(payload 為 dict)。"""

    @component.output_types(payload=dict[str, Any])
    def run(self, documents: list[Document], query: str) -> dict[str, Any]:
        return {"payload": {
            "question": query,
            "ids": [d.meta.get("chunk_id") for d in documents],
        }}
'''

FORMATTER_STR_PAYLOAD = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class CsvFormatter:
    """payload 型別自由:宣告 str 也要能過契約(終端槽位的 Any 特判)。"""

    @component.output_types(payload=str)
    def run(self, documents: list[Document], query: str) -> dict[str, Any]:
        return {"payload": "\\n".join(d.meta.get("chunk_id", "") for d in documents)}
'''

FORMATTER_MISSING_QUERY = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class NoQueryFormatter:
    @component.output_types(payload=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        return {"payload": {}}
'''

FORMATTER_WRONG_OUTPUT_NAME = '''
from typing import Any
from haystack import component
from haystack.dataclasses import Document


@component
class WrongOutputFormatter:
    """輸入正確,但輸出 socket 名稱不是 payload。"""

    @component.output_types(result=dict[str, Any])
    def run(self, documents: list[Document], query: str) -> dict[str, Any]:
        return {"result": {}}
'''


class TestCustomFormatter:
    """formatter 槽位:documents + query → payload: Any(終端支線)。"""

    @staticmethod
    def _config(corpus_dir, formatter):
        data = make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                }
            },
            inference={"reranking": {"method": "none"}},
        )
        data["inference"]["formatter"] = formatter
        return parse_config(data)

    def test_custom_formatter_end_to_end(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, FORMATTER_OK)
        config = self._config(
            corpus_dir,
            {"method": "custom", "params": {"file": file, "class": "EnvelopeFormatter"}},
        )
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()

        result = pipelines.query("向量檢索")
        # payload 進固定的 output 鍵;canonical 鍵照舊
        assert result["output"]["question"] == "向量檢索"
        assert result["output"]["ids"], "信封應含檢索到的 chunk_id"
        assert result["documents"], "canonical documents 不受 formatter 影響"
        # trace 同步收錄
        assert result["trace"]["formatter"]["type"] == "EnvelopeFormatter"
        assert result["trace"]["formatter"]["payload"] == result["output"]

    def test_payload_type_is_free(self, tmp_path, corpus_dir):
        """終端槽位特權:元件宣告 payload=str 也能過契約與端到端。"""
        file = write_module(tmp_path, FORMATTER_STR_PAYLOAD)
        config = self._config(
            corpus_dir,
            {"method": "custom", "params": {"file": file, "class": "CsvFormatter"}},
        )
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()
        result = pipelines.query("向量檢索")
        assert isinstance(result["output"], str)
        assert "chunk_0" in result["output"]

    def test_missing_query_input_rejected(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, FORMATTER_MISSING_QUERY)
        config = self._config(
            corpus_dir,
            {"method": "custom", "params": {"file": file, "class": "NoQueryFormatter"}},
        )
        with pytest.raises(ConfigError, match="缺少輸入 socket 'query"):
            build_pipelines(config)

    def test_missing_payload_output_rejected(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, FORMATTER_WRONG_OUTPUT_NAME)
        config = self._config(
            corpus_dir,
            {
                "method": "custom",
                "params": {"file": file, "class": "WrongOutputFormatter"},
            },
        )
        with pytest.raises(ConfigError, match="缺少輸出 socket 'payload"):
            build_pipelines(config)


class TestCustomImport:
    """import 槽位:無輸入 → sources + meta(必帶 doc_id)。"""

    @staticmethod
    def _config(import_params, parsing=None):
        return parse_config(
            make_config(
                ingestion={
                    "import": {"method": "custom", "params": import_params},
                    "parsing": parsing or {
                        "method": "custom",
                        "params": {},  # 由測試填
                    },
                },
                inference={"reranking": {"method": "none"}},
            )
        )

    def test_bytestream_end_to_end(self, tmp_path):
        """ByteStream 來源(不落地)一路到穩定 chunk_id。"""
        importer = write_module(tmp_path, IMPORTER_BYTESTREAM_OK, "importer.py")
        converter = write_module(tmp_path, CONVERTER_OK, "converter.py")
        config = self._config(
            {"file": importer, "class": "FakeApiImporter", "content_type": "text"},
            parsing={
                "method": "custom",
                "params": {
                    "file": converter, "class": "EnvelopeParser", "kind": "converter"
                },
            },
        )
        pipelines = build_pipelines(config)
        result = pipelines.run_ingestion()
        written = {doc.id for doc in pipelines.store.filter_documents()}
        assert "API-1::chunk_0" in written and "API-2::chunk_0" in written
        assert result["writer"]["documents_written"] == 2

        answer = pipelines.query("請假")
        assert answer["documents"][0].meta["doc_id"] == "API-1"

    def test_mandatory_input_rejected(self, tmp_path):
        file = write_module(tmp_path, IMPORTER_NEEDS_INPUT)
        config = self._config({"file": file, "class": "NeedsQueryImporter"})
        with pytest.raises(ConfigError) as excinfo:
            build_pipelines(config)
        assert "不提供任何輸入" in str(excinfo.value)

    def test_content_type_feeds_compatibility_check(self, tmp_path):
        file = write_module(tmp_path, IMPORTER_BYTESTREAM_OK)
        config = self._config(
            {"file": file, "class": "FakeApiImporter", "content_type": "pdf"},
            parsing={"method": "plain_text"},
        )
        from rag.errors import IncompatiblePipelineError

        with pytest.raises(IncompatiblePipelineError, match="'pdf'"):
            build_pipelines(config)

    def test_no_content_type_skips_check(self, tmp_path):
        file = write_module(tmp_path, IMPORTER_BYTESTREAM_OK)
        config = self._config(
            {"file": file, "class": "FakeApiImporter"},
            parsing={"method": "plain_text"},
        )
        build_pipelines(config)  # 未宣告 → 不做相容性檢查,建構成功

    def test_missing_doc_id_dies_at_stamper_with_pointer(self, tmp_path):
        file = write_module(tmp_path, IMPORTER_NO_DOC_ID)
        converter = write_module(tmp_path, CONVERTER_OK, "converter.py")
        config = self._config(
            {"file": file, "class": "NoDocIdImporter"},
            parsing={
                "method": "custom",
                "params": {
                    "file": converter, "class": "EnvelopeParser", "kind": "converter"
                },
            },
        )
        pipelines = build_pipelines(config)
        from haystack.core.errors import PipelineRuntimeError

        # Haystack 會把元件錯誤包成 PipelineRuntimeError;訊息保留可行動指引
        with pytest.raises(PipelineRuntimeError, match="doc_id"):
            pipelines.run_ingestion()


class TestCustomParsing:
    """parsing 槽位:kind 決定契約(converter / doc_processor)與鏈位置。"""

    @staticmethod
    def _config(corpus_dir, parsing):
        return parse_config(
            make_config(
                ingestion={
                    "import": {
                        "method": "local_file",
                        "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                    },
                    "parsing": parsing,
                },
                inference={"reranking": {"method": "none"}},
            )
        )

    def test_doc_processor_in_chain(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, DOC_PROCESSOR_OK)
        config = self._config(
            corpus_dir,
            {
                "method": ["plain_text", "custom"],
                "method_params": {
                    "custom": {"file": file, "class": "UpperCaser"},
                },
            },
        )
        pipelines = build_pipelines(config)
        assert type(
            pipelines.ingestion.get_component("parser_2")
        ).__name__ == "UpperCaser"
        pipelines.run_ingestion()
        docs = pipelines.store.filter_documents()
        assert docs
        # 語料原文是 "Facebook":大寫化生效才會變 FACEBOOK
        assert any("FACEBOOK" in (d.content or "") for d in docs), "大寫化應已生效"

    def test_default_kind_rejected_at_chain_head(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, DOC_PROCESSOR_OK)
        config = self._config(
            corpus_dir,
            {"method": "custom", "params": {"file": file, "class": "UpperCaser"}},
        )
        with pytest.raises(ConfigError, match="kind: converter"):
            build_pipelines(config)

    def test_converter_kind_rejected_mid_chain(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, CONVERTER_OK)
        config = self._config(
            corpus_dir,
            {
                "method": ["plain_text", "custom"],
                "method_params": {
                    "custom": {
                        "file": file, "class": "EnvelopeParser", "kind": "converter"
                    },
                },
            },
        )
        with pytest.raises(ConfigError, match="converter 只能放在鏈首"):
            build_pipelines(config)

    def test_narrow_sources_annotation_rejected(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, CONVERTER_NARROW_SOURCES)
        config = self._config(
            corpus_dir,
            {
                "method": "custom",
                "params": {
                    "file": file, "class": "PathOnlyParser", "kind": "converter"
                },
            },
        )
        with pytest.raises(ConfigError) as excinfo:
            build_pipelines(config)
        assert "ByteStream" in str(excinfo.value)  # 訊息點名期望的 union

    def test_produces_pages_satisfies_page_based(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, CONVERTER_OK)
        config = self._config(
            corpus_dir,
            {
                "method": "custom",
                "params": {
                    "file": file, "class": "EnvelopeParser",
                    "kind": "converter", "produces_pages": True,
                },
            },
        )
        config.ingestion.chunking.method = "page_based"
        build_pipelines(config)  # 不宣告 produces_pages 的話這裡會不相容

    def test_input_content_types_feeds_compatibility(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, CONVERTER_OK)
        config = self._config(
            corpus_dir,
            {
                "method": "custom",
                "params": {
                    "file": file, "class": "EnvelopeParser",
                    "kind": "converter", "input_content_types": ["pdf"],
                },
            },
        )
        from rag.errors import IncompatiblePipelineError

        with pytest.raises(IncompatiblePipelineError, match="'text'"):
            build_pipelines(config)

    def test_input_content_types_on_doc_processor_rejected(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, DOC_PROCESSOR_OK)
        config = self._config(
            corpus_dir,
            {
                "method": ["plain_text", "custom"],
                "method_params": {
                    "custom": {
                        "file": file, "class": "UpperCaser",
                        "input_content_types": ["text"],
                    },
                },
            },
        )
        with pytest.raises(ConfigError, match="只對 kind: converter 有意義"):
            build_pipelines(config)


class TestCustomChunking:
    """chunking 槽位:documents → documents;requires_pages 宣告進相容性檢查。"""

    @staticmethod
    def _config(corpus_dir, chunking_params):
        return parse_config(
            make_config(
                ingestion={
                    "import": {
                        "method": "local_file",
                        "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                    },
                    "chunking": {"method": "custom", "params": chunking_params},
                },
                inference={"reranking": {"method": "none"}},
            )
        )

    def test_custom_chunker_end_to_end(self, tmp_path, corpus_dir):
        file = write_module(tmp_path, CHUNKER_OK)
        config = self._config(corpus_dir, {"file": file, "class": "LineChunker"})
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()
        docs = pipelines.store.filter_documents()
        assert docs
        # 一行一塊 + stamper 蓋章:同一文件的切片 seq 連號
        faiss_chunks = sorted(
            d.id for d in docs if d.meta["doc_id"] == "faiss.txt"
        )
        assert faiss_chunks[0] == "faiss.txt::chunk_0"
        assert len(faiss_chunks) >= 2, "兩段內容應切成至少兩塊"

    def test_requires_pages_rejected_without_paginated_parser(
        self, tmp_path, corpus_dir
    ):
        file = write_module(tmp_path, CHUNKER_OK)
        config = self._config(
            corpus_dir,
            {"file": file, "class": "LineChunker", "requires_pages": True},
        )
        from rag.errors import IncompatiblePipelineError

        with pytest.raises(IncompatiblePipelineError, match="需要分頁輸入"):
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

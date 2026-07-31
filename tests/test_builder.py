"""builder 測試:未知方法、方法鏈規則、參數驗證、no_chunking。"""

from __future__ import annotations

import pytest
from conftest import make_config

from rag.builder import build_ingestion_pipeline, build_pipelines
from rag.config import parse_config
from rag.errors import ConfigError, UnknownMethodError


def test_unknown_method_lists_alternatives():
    config = parse_config(
        make_config(ingestion={"chunking": {"method": "sementic"}})
    )
    with pytest.raises(UnknownMethodError, match="'fixed_size'") as excinfo:
        build_ingestion_pipeline(config)
    message = str(excinfo.value)
    assert "模組 'chunking' 沒有名為 'sementic' 的方法" in message
    assert "'page_based'" in message and "'no_chunking'" in message


def test_non_chainable_slot_rejects_method_list():
    config = parse_config(
        make_config(
            ingestion={
                "chunking": {
                    "method": ["fixed_size", "page_based"],
                    "method_params": {},
                }
            }
        )
    )
    with pytest.raises(ConfigError, match="不支援方法鏈"):
        build_ingestion_pipeline(config)


def test_parsing_chain_head_must_be_converter():
    config = parse_config(make_config(ingestion={"parsing": {"method": "clean"}}))
    with pytest.raises(ConfigError, match="鏈的第一個方法必須是 converter"):
        build_ingestion_pipeline(config)


def test_parsing_chain_converter_only_at_head():
    config = parse_config(
        make_config(
            ingestion={
                "parsing": {"method": ["plain_text", "pdf"], "method_params": {}}
            }
        )
    )
    with pytest.raises(ConfigError, match="converter 只能放在鏈首"):
        build_ingestion_pipeline(config)


def test_invalid_params_list_accepted_fields():
    config = parse_config(
        make_config(
            ingestion={
                "chunking": {"method": "fixed_size", "params": {"chunk_size": 128}}
            }
        )
    )
    with pytest.raises(ConfigError, match="可接受的參數:.*split_length"):
        build_ingestion_pipeline(config)


def test_param_cross_constraint_message():
    config = parse_config(
        make_config(
            ingestion={
                "chunking": {
                    "method": "fixed_size",
                    "params": {"split_length": 10, "split_overlap": 10},
                }
            }
        )
    )
    with pytest.raises(ConfigError, match="split_overlap 必須小於 split_length"):
        build_ingestion_pipeline(config)


def test_no_chunking_skips_splitter(corpus_dir):
    config = parse_config(
        make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir)},
                },
                "chunking": {"method": "no_chunking"},
            }
        )
    )
    pipeline, store = build_ingestion_pipeline(config)
    assert "chunker" not in pipeline.to_dict()["components"]
    pipeline.run({})
    docs = store.filter_documents()
    # 每份文件恰好一個切片(seq 固定為 0)
    assert sorted(d.id for d in docs) == ["faiss.txt::chunk_0", "sub/es.txt::chunk_0"]


def test_native_pipeline_stage_requires_build_pipelines():
    data = make_config()
    del data["ingestion"]
    data["haystack_pipelines"] = {"ingestion": "pipelines/native.yaml"}
    config = parse_config(data)
    with pytest.raises(ConfigError, match="build_pipelines"):
        build_ingestion_pipeline(config)


class TestEscapeHatch:
    def _native_inference_config(self, tmp_path, corpus_dir):
        from haystack import Pipeline
        from haystack.components.builders import PromptBuilder

        native = Pipeline()
        native.add_component("prompt_builder", PromptBuilder(template="Q: {{ q }}"))
        path = tmp_path / "native_inference.yaml"
        path.write_text(native.dumps(), encoding="utf-8")

        data = make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir)},
                }
            }
        )
        del data["inference"]
        data["haystack_pipelines"] = {"inference": str(path)}
        return parse_config(data)

    def test_native_inference_loads_and_disables_query_helper(
        self, tmp_path, corpus_dir
    ):
        config = self._native_inference_config(tmp_path, corpus_dir)
        pipelines = build_pipelines(config)
        assert pipelines.query_entry is None
        # 原生 pipeline 本身可直接執行
        out = pipelines.inference.run({"prompt_builder": {"q": "hi"}})
        assert out["prompt_builder"]["prompt"] == "Q: hi"
        # query() 便利介面明確拒絕並指路
        with pytest.raises(ConfigError, match="原生 Haystack YAML"):
            pipelines.query("hi")

    def test_native_pipeline_missing_file(self, corpus_dir):
        data = make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir)},
                }
            }
        )
        del data["inference"]
        data["haystack_pipelines"] = {"inference": "does/not/exist.yaml"}
        with pytest.raises(ConfigError, match="找不到 haystack_pipelines 指定的檔案"):
            build_pipelines(parse_config(data))

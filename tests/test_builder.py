"""builder 測試:未知方法、方法鏈規則、參數驗證、no_chunking、
ES 認證欄位、boost_k_factor、檢索-only(generate_answer: false)。"""

from __future__ import annotations

import pytest
from conftest import make_config

from rag.builder import (
    build_inference_pipeline,
    build_ingestion_pipeline,
    build_pipelines,
)
from rag.components.api_clients import FlexibleAPIRanker
from rag.methods_ingestion import (
    _elasticsearch_client_kwargs,
    _ElasticsearchParams,
    _elasticsearch_auth_kwargs,
    _elasticsearch_store_kwargs,
)
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
                    "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
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


class TestElasticsearchAuth:
    """ES 認證欄位:半套 / 衝突的組合在建 pipeline 時就擋下,不拖到送請求。"""

    @staticmethod
    def _config(**params):
        return parse_config(
            make_config(
                ingestion={
                    "indexing": {
                        "method": "elasticsearch",
                        "params": {"hosts": "https://es.example.com:9200", **params},
                    }
                }
            )
        )

    def test_username_without_password_fails(self):
        config = self._config(username="elastic")
        with pytest.raises(ConfigError, match="必須成對提供,目前缺少 password"):
            build_ingestion_pipeline(config)

    def test_password_without_username_fails(self):
        config = self._config(password="secret")
        with pytest.raises(ConfigError, match="必須成對提供,目前缺少 username"):
            build_ingestion_pipeline(config)

    def test_api_key_and_basic_auth_are_mutually_exclusive(self):
        config = self._config(api_key="tok", username="elastic", password="secret")
        with pytest.raises(ConfigError, match="只能擇一"):
            build_ingestion_pipeline(config)

    def test_basic_auth_becomes_two_item_tuple(self):
        # 字串形式("user:pass")會被 elasticsearch client 當成已編碼的值
        # 原樣送出,認證必失敗 —— 這裡釘住 2 元組。
        params = _ElasticsearchParams(
            hosts="https://es.example.com:9200", username="elastic", password="secret"
        )
        assert _elasticsearch_auth_kwargs(params) == {
            "basic_auth": ("elastic", "secret")
        }

    def test_api_key_passed_through(self):
        params = _ElasticsearchParams(hosts="https://es.example.com:9200", api_key="tok")
        assert _elasticsearch_auth_kwargs(params) == {"api_key": "tok"}

    def test_no_credentials_stays_empty(self):
        # 本地 docker-compose 的 ES 關掉了 security,不帶憑證是合法的。
        params = _ElasticsearchParams(hosts="http://localhost:9200")
        assert _elasticsearch_auth_kwargs(params) == {}

    def test_unknown_auth_field_lists_accepted_params(self):
        config = self._config(basic_auth=["elastic", "secret"])
        with pytest.raises(ConfigError, match="可接受的參數:.*username"):
            build_ingestion_pipeline(config)


class TestElasticsearchStoreKwargs:
    """store 建構參數的組裝:選填欄位不設定就不帶,設定了原樣透傳。"""

    def test_defaults_omit_optional_fields(self):
        params = _ElasticsearchParams(hosts="http://localhost:9200")
        kwargs = _elasticsearch_store_kwargs(params)
        assert kwargs == {
            "hosts": "http://localhost:9200",
            "index": "modular-rag",
            "embedding_similarity_function": "cosine",
        }

    def test_custom_mapping_and_ingest_pipeline_passed_through(self):
        mapping = {
            "properties": {
                "content": {"type": "text", "analyzer": "ik_smart"},
                "embedding": {"type": "dense_vector", "dims": 384},
            }
        }
        params = _ElasticsearchParams(
            hosts="http://localhost:9200",
            custom_mapping=mapping,
            ingest_pipeline="company-pipeline",
        )
        kwargs = _elasticsearch_store_kwargs(params)
        assert kwargs["custom_mapping"] == mapping
        assert kwargs["ingest_pipeline"] == "company-pipeline"

    def test_settings_and_fields_not_leaked_to_store(self):
        # settings 由框架預建索引使用、fields 由 MetaFieldMapper 使用,
        # 都不是 ElasticsearchDocumentStore 的建構參數。
        params = _ElasticsearchParams(
            hosts="http://localhost:9200",
            custom_mapping={"properties": {}},
            settings={"analysis": {}},
            fields={"summary_text": "summary"},
        )
        kwargs = _elasticsearch_store_kwargs(params)
        assert "settings" not in kwargs and "fields" not in kwargs

    def test_request_timeout_passed_through(self):
        """store 未列名的參數會原樣轉給 elasticsearch client(bulk 逾時用)。"""
        params = _ElasticsearchParams(
            hosts="http://localhost:9200", request_timeout=60
        )
        assert _elasticsearch_store_kwargs(params)["request_timeout"] == 60

    def test_retry_params_passed_through(self):
        params = _ElasticsearchParams(
            hosts="http://localhost:9200", retry_on_timeout=True, max_retries=5
        )
        kwargs = _elasticsearch_store_kwargs(params)
        assert kwargs["retry_on_timeout"] is True
        assert kwargs["max_retries"] == 5

    def test_client_kwargs_shared_with_prebuild_path(self):
        """settings 預建索引的 client 必須拿到與 store 相同的 timeout / 重試
        設定 —— 否則預建的 HEAD 用 client 預設(10 秒、不重試)打出去,
        調了 request_timeout 也治不到它。"""
        params = _ElasticsearchParams(
            hosts="https://es.example.com:9200",
            api_key="tok",
            ca_certs="/certs/ca.crt",
            request_timeout=60,
            retry_on_timeout=True,
            max_retries=5,
        )
        assert _elasticsearch_client_kwargs(params) == {
            "api_key": "tok",
            "ca_certs": "/certs/ca.crt",
            "request_timeout": 60,
            "retry_on_timeout": True,
            "max_retries": 5,
        }

    def test_client_kwargs_defaults_stay_empty(self):
        # 不設定就不帶:交給 elasticsearch client 用自己的預設值。
        params = _ElasticsearchParams(hosts="http://localhost:9200")
        assert _elasticsearch_client_kwargs(params) == {}

    def test_max_retries_must_be_at_least_one(self):
        with pytest.raises(ConfigError, match="max_retries"):
            build_ingestion_pipeline(
                parse_config(
                    make_config(
                        ingestion={
                            "indexing": {
                                "method": "elasticsearch",
                                "params": {
                                    "hosts": "http://localhost:9200",
                                    "max_retries": 0,
                                },
                            }
                        }
                    )
                )
            )

    def test_request_timeout_must_be_positive(self):
        with pytest.raises(ConfigError, match="request_timeout"):
            build_ingestion_pipeline(
                parse_config(
                    make_config(
                        ingestion={
                            "indexing": {
                                "method": "elasticsearch",
                                "params": {
                                    "hosts": "http://localhost:9200",
                                    "request_timeout": 0,
                                },
                            }
                        }
                    )
                )
            )


class TestElasticsearchIndexSettings:
    """settings 預建索引:需搭配 custom_mapping;已存在的索引不動。"""

    def test_settings_without_custom_mapping_rejected(self):
        config = parse_config(
            make_config(
                ingestion={
                    "indexing": {
                        "method": "elasticsearch",
                        "params": {
                            "hosts": "http://localhost:9200",
                            "settings": {"analysis": {}},
                        },
                    }
                }
            )
        )
        with pytest.raises(ConfigError, match="settings 需要搭配 custom_mapping"):
            build_ingestion_pipeline(config)

    class _StubClient:
        def __init__(self, exists: bool) -> None:
            self._exists = exists
            self.created: dict | None = None
            outer = self

            class _Indices:
                def exists(self, index):
                    return outer._exists

                def create(self, index, mappings, settings):
                    outer.created = {
                        "index": index, "mappings": mappings, "settings": settings
                    }

            self.indices = _Indices()

    def test_ensure_es_index_creates_when_missing(self):
        from rag.methods_ingestion import _ensure_es_index

        params = _ElasticsearchParams(
            hosts="http://localhost:9200",
            index="kb",
            custom_mapping={"properties": {}},
            settings={"analysis": {"analyzer": {}}},
        )
        client = self._StubClient(exists=False)
        _ensure_es_index(client, params)
        assert client.created == {
            "index": "kb",
            "mappings": {"properties": {}},
            "settings": {"analysis": {"analyzer": {}}},
        }

    def test_ensure_es_index_skips_existing(self):
        from rag.methods_ingestion import _ensure_es_index

        params = _ElasticsearchParams(
            hosts="http://localhost:9200",
            custom_mapping={"properties": {}},
            settings={},
        )
        client = self._StubClient(exists=True)
        _ensure_es_index(client, params)
        assert client.created is None


class TestAPIRerankSlot:
    """api_rerank 走完整的 config → pipeline 路徑(元件行為見 test_api_reranker)。"""

    def test_built_into_reranking_chain(self):
        config = parse_config(
            make_config(
                inference={
                    "reranking": {
                        "method": "api_rerank",
                        "params": {
                            "endpoint": "https://rerank.example.com/v1/rerank",
                            "headers": {"Authorization": "Bearer T"},
                            "model": "rerank-v1",
                            "top_k": 3,
                        },
                    }
                }
            )
        )
        outer, _meta = build_inference_pipeline(config)
        # ranker 在 MultiQueryRetrievalStage 的內部 pipeline 裡(每個子查詢各跑一次)
        ranker = outer.get_component("multi_query").inner.get_component("ranker")
        assert isinstance(ranker, FlexibleAPIRanker)
        assert ranker.top_k == 3 and ranker.model == "rerank-v1"
        assert ranker.headers == {"Authorization": "Bearer T"}

    def test_endpoint_is_required(self):
        config = parse_config(
            make_config(
                inference={"reranking": {"method": "api_rerank", "params": {"top_k": 3}}}
            )
        )
        with pytest.raises(ConfigError, match="endpoint"):
            build_inference_pipeline(config)

    def test_index_base_must_be_zero_or_one(self):
        config = parse_config(
            make_config(
                inference={
                    "reranking": {
                        "method": "api_rerank",
                        "params": {
                            "endpoint": "https://rerank.example.com/v1/rerank",
                            "index_base": 2,
                        },
                    }
                }
            )
        )
        with pytest.raises(ConfigError, match="index_base"):
            build_inference_pipeline(config)

    def test_listed_as_available_method(self):
        config = parse_config(
            make_config(inference={"reranking": {"method": "api_reranking"}})
        )
        with pytest.raises(UnknownMethodError, match="'api_rerank'"):
            build_inference_pipeline(config)


class TestBoostKFactor:
    def test_bm25_retriever_fetches_boosted_top_k(self):
        config = parse_config(
            make_config(
                inference={
                    "retrieval": {
                        "method": "bm25",
                        "params": {"top_k": 2, "boost_k_factor": 3},
                    }
                }
            )
        )
        pipeline, _ = build_inference_pipeline(config)
        inner = pipeline.get_component("multi_query").inner
        assert inner.get_component("retriever").top_k == 6

    def test_hybrid_boosts_both_retrievers_and_joiner(self):
        config = parse_config(
            make_config(
                inference={
                    "retrieval": {
                        "method": "hybrid",
                        "params": {"top_k": 4, "boost_k_factor": 3},
                    }
                }
            )
        )
        pipeline, _ = build_inference_pipeline(config)
        inner = pipeline.get_component("multi_query").inner
        assert inner.get_component("bm25_retriever").top_k == 12
        assert inner.get_component("embedding_retriever").top_k == 12
        assert inner.get_component("joiner").top_k == 12

    def test_default_factor_keeps_top_k(self):
        config = parse_config(
            make_config(inference={"retrieval": {"method": "bm25", "params": {"top_k": 5}}})
        )
        pipeline, _ = build_inference_pipeline(config)
        inner = pipeline.get_component("multi_query").inner
        assert inner.get_component("retriever").top_k == 5

    def test_zero_factor_rejected(self):
        config = parse_config(
            make_config(
                inference={
                    "retrieval": {
                        "method": "bm25",
                        "params": {"top_k": 5, "boost_k_factor": 0},
                    }
                }
            )
        )
        with pytest.raises(ConfigError, match="boost_k_factor"):
            build_inference_pipeline(config)


class TestFactCheckRegistration:
    def test_unknown_rerank_method_lists_fact_check(self):
        config = parse_config(
            make_config(inference={"reranking": {"method": "factcheck"}})
        )
        with pytest.raises(UnknownMethodError, match="'llm_fact_check'"):
            build_inference_pipeline(config)

    def test_chained_after_llm_rerank(self):
        config = parse_config(
            make_config(
                inference={
                    "reranking": {
                        "method": ["llm", "llm_fact_check"],
                        "method_params": {},
                    }
                }
            )
        )
        pipeline, _ = build_inference_pipeline(config)
        inner = pipeline.get_component("multi_query").inner
        from rag.components.fact_check import LLMFactChecker

        assert inner.get_component("ranker") is not None
        assert isinstance(inner.get_component("ranker_2"), LLMFactChecker)


class TestResearchMethodRegistration:
    """preqrag / llm_multi_hyde / insertrank 走完整的 config → pipeline
    路徑(元件行為見 test_query_transforms / test_llm_rerankers)。"""

    def test_preqrag_built_with_params(self):
        from rag.components.query_transforms import PreQRAGDispatcher

        config = parse_config(
            make_config(
                inference={
                    "query_transformation": {
                        "method": "preqrag",
                        "params": {"num_rewrites": 3, "include_original": False},
                    }
                }
            )
        )
        pipeline, _ = build_inference_pipeline(config)
        dispatcher = pipeline.get_component("transform")
        assert isinstance(dispatcher, PreQRAGDispatcher)
        assert dispatcher.num_rewrites == 3
        assert dispatcher.include_original is False

    def test_multi_hyde_built_with_params(self):
        from rag.components.query_transforms import LLMMultiHyDEExpander

        config = parse_config(
            make_config(
                inference={
                    "query_transformation": {
                        "method": "llm_multi_hyde",
                        "params": {"num_documents": 2, "keep_original": False},
                    }
                }
            )
        )
        pipeline, _ = build_inference_pipeline(config)
        expander = pipeline.get_component("transform")
        assert isinstance(expander, LLMMultiHyDEExpander)
        assert expander.num_documents == 2
        assert expander.keep_original is False

    def test_insertrank_built_with_params(self):
        from rag.components.llm_rerankers import InsertRankLLMRanker

        config = parse_config(
            make_config(
                inference={
                    "reranking": {
                        "method": "insertrank",
                        "params": {"top_k": 3, "score_label": "BM25 分數"},
                    }
                }
            )
        )
        pipeline, _ = build_inference_pipeline(config)
        ranker = pipeline.get_component("multi_query").inner.get_component("ranker")
        assert isinstance(ranker, InsertRankLLMRanker)
        assert (ranker.top_k, ranker.score_label) == (3, "BM25 分數")


class TestSkipGeneration:
    def test_missing_generation_with_default_flag_rejected(self):
        data = make_config()
        del data["inference"]["generation"]
        with pytest.raises(ConfigError, match="generate_answer"):
            parse_config(data)

    def test_flag_false_omits_generator_and_prompt_builder(self):
        config = parse_config(
            make_config(inference={"generate_answer": False})
        )
        pipeline, meta = build_inference_pipeline(config)
        components = pipeline.to_dict()["components"]
        assert "generator" not in components
        assert "prompt_builder" not in components
        assert meta["generate_answer"] is False

    def test_flag_false_without_generation_requires_explicit_generator(self):
        # reranking 為 llm 且未指定 generator:沒有 generation 槽位可沿用
        data = make_config(inference={"generate_answer": False})
        del data["inference"]["generation"]
        config = parse_config(data)
        with pytest.raises(ConfigError, match="需要 chat generator"):
            build_inference_pipeline(config)

    def test_flag_false_with_generation_block_still_inherits(self):
        # generation 區塊保留(mock)時,llm rerank 沿用成功
        config = parse_config(
            make_config(inference={"generate_answer": False})
        )
        pipeline, _ = build_inference_pipeline(config)
        inner = pipeline.get_component("multi_query").inner
        assert "ranker" in inner.to_dict()["components"]


class TestStageSelection:
    """``stage=…``:只組其中一段(run_demo.py / serve.py 的 --stage)。"""

    @staticmethod
    def _config(corpus_dir):
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
                }
            )
        )

    def test_only_ingestion_is_built(self, corpus_dir):
        pipelines = build_pipelines(self._config(corpus_dir), stage="ingestion")
        assert pipelines.inference is None
        assert pipelines.query_entry is None
        # 索引照建(store 由 ingestion 側建立並回收)
        assert pipelines.run_ingestion()["writer"]["documents_written"] > 0
        assert pipelines.store.count_documents() > 0

    def test_query_refuses_and_points_at_fix(self, corpus_dir):
        pipelines = build_pipelines(self._config(corpus_dir), stage="ingestion")
        with pytest.raises(ConfigError, match="stage='ingestion'"):
            pipelines.query("向量檢索")

    def test_unknown_stage_lists_options(self, corpus_dir):
        with pytest.raises(ConfigError, match="'all', 'ingestion', 'inference'"):
            build_pipelines(self._config(corpus_dir), stage="ingest")

    def test_only_inference_is_built(self, corpus_dir):
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
                }
            )
        )
        pipelines = build_pipelines(config, stage="inference")
        assert pipelines.ingestion is None
        assert pipelines.store is not None  # inference 端自行依 indexing 建
        assert pipelines.query_entry is not None
        # 查詢仍可執行(索引是空的,結果就是空的 —— 不是錯誤)
        assert pipelines.query("向量檢索")["documents"] == []

    def test_run_ingestion_refuses_and_points_at_fix(self, corpus_dir):
        config = parse_config(make_config())
        pipelines = build_pipelines(config, stage="inference")
        with pytest.raises(ConfigError, match="stage='inference'"):
            pipelines.run_ingestion()

    def test_missing_source_dir_is_not_touched(self, tmp_path):
        """來源資料夾不必存在:ingestion 側元件根本沒建。"""
        config = parse_config(
            make_config(
                ingestion={
                    "import": {
                        "method": "local_file",
                        "params": {
                            "input_dir": str(tmp_path / "no_such_dir"),
                            "extensions": [".txt"],
                        },
                    }
                }
            )
        )
        pipelines = build_pipelines(config, stage="inference")
        assert pipelines.ingestion is None

    def test_default_still_builds_ingestion(self, corpus_dir):
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
                }
            )
        )
        pipelines = build_pipelines(config)
        assert pipelines.ingestion is not None
        assert pipelines.run_ingestion()["writer"]["documents_written"] > 0


class TestFormatterSlot:
    """formatter:選填終端支線(simple_json 內建方法與省略行為)。"""

    @staticmethod
    def _config(corpus_dir, formatter=None, **inference_overrides):
        data = make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                }
            },
            inference={"reranking": {"method": "none"}, **inference_overrides},
        )
        if formatter is not None:
            data["inference"]["formatter"] = formatter
        return parse_config(data)

    def test_omitted_slot_means_no_component_and_none_output(self, corpus_dir):
        pipelines = build_pipelines(self._config(corpus_dir))
        assert "formatter" not in pipelines.inference.to_dict()["components"]
        pipelines.run_ingestion()
        result = pipelines.query("向量檢索")
        assert result["output"] is None
        assert result["trace"]["formatter"] is None

    def test_simple_json_shape(self, corpus_dir):
        pipelines = build_pipelines(
            self._config(corpus_dir, formatter={"method": "simple_json"})
        )
        pipelines.run_ingestion()
        result = pipelines.query("向量檢索")

        payload = result["output"]
        assert payload["query"] == "向量檢索"
        assert payload["total"] == len(payload["documents"])
        first = payload["documents"][0]
        assert set(first) == {"doc_id", "chunk_id", "page", "score", "content"}
        assert first["chunk_id"] == result["documents"][0].meta["chunk_id"]

    def test_simple_json_without_content(self, corpus_dir):
        pipelines = build_pipelines(
            self._config(
                corpus_dir,
                formatter={
                    "method": "simple_json",
                    "params": {"include_content": False},
                },
            )
        )
        pipelines.run_ingestion()
        payload = pipelines.query("向量檢索")["output"]
        assert all("content" not in row for row in payload["documents"])

    def test_coexists_with_generation(self, corpus_dir):
        """開生成時 answer 與 output 並存(fusion.documents fan-out 兩支線)。"""
        data = make_config(
            ingestion={
                "import": {
                    "method": "local_file",
                    "params": {"input_dir": str(corpus_dir), "extensions": [".txt"]},
                }
            },
            inference={
                "reranking": {"method": "none"},
                "generation": {"method": "mock", "params": {"replies": ["答案"]}},
            },
        )
        data["inference"]["formatter"] = {"method": "simple_json"}
        pipelines = build_pipelines(parse_config(data))
        pipelines.run_ingestion()
        result = pipelines.query("向量檢索")
        assert result["answer"] == "答案"
        assert result["output"]["total"] >= 1

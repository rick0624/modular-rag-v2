"""routing 槽位:內建 keyword_match、custom、trace 排版、服務端到端。"""

from __future__ import annotations

import copy

import pytest
import yaml
from conftest import BASE_CONFIG, make_config

from rag.builder import build_pipelines
from rag.config import parse_config
from rag.errors import ConfigError, UnknownMethodError
from rag.trace import format_query_trace

ROUTES = {
    "人資/差勤": ["請假", "特休", "薪資"],
    "資訊系統": ["VPN", "密碼", "登入"],
}

ROUTING_KEYWORD = {
    "method": "keyword_match",
    "params": {"routes": ROUTES, "default_category": "其他"},
}

ROUTER_OK = '''
from typing import Any
from haystack import component


@component
class FixedRouter:
    @component.output_types(route=dict[str, Any])
    def run(self, query: str) -> dict[str, Any]:
        return {"route": {"category": "測試類別", "source": "custom"}}
'''

ROUTER_BAD_OUTPUT = '''
from typing import Any
from haystack import component


@component
class BadRouter:
    @component.output_types(result=dict[str, Any])
    def run(self, query: str) -> dict[str, Any]:
        return {"result": {}}
'''


def routing_config(corpus_dir, routing: dict):
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
            inference={"routing": routing},
        )
    )


class TestRoutingSlot:
    def test_routing_absent_by_default(self, corpus_dir):
        config = routing_config(corpus_dir, None)
        # make_config 會把 None 覆寫進去,等同 config 省略 routing
        pipelines = build_pipelines(config)
        assert pipelines.routing_enabled is False
        assert "routing" not in pipelines.inference.graph.nodes
        pipelines.run_ingestion()
        result = pipelines.query("如何請假?")
        assert result["routing"] is None
        assert result["trace"]["routing"] is None

    def test_keyword_match_end_to_end(self, corpus_dir):
        config = routing_config(corpus_dir, ROUTING_KEYWORD)
        pipelines = build_pipelines(config)
        assert pipelines.routing_enabled is True
        pipelines.run_ingestion()
        result = pipelines.query("請假要在差勤系統怎麼申請?")
        route = result["routing"]
        assert route["category"] == "人資/差勤"
        assert route["matched_keywords"] == ["請假"]
        assert result["trace"]["routing"] == route
        # 檢索與生成不受影響
        assert result["answer"] is not None

    def test_keyword_match_default_category(self, corpus_dir):
        config = routing_config(corpus_dir, ROUTING_KEYWORD)
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()
        result = pipelines.query("向量檢索是什麼?")
        assert result["routing"]["category"] == "其他"
        assert result["routing"]["confidence"] == 0.0

    def test_routing_with_retrieval_only(self, corpus_dir):
        data = make_config(
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
                "routing": ROUTING_KEYWORD,
                "generate_answer": False,
                "reranking": {"method": "none"},  # llm 重排需沿用 generation
            },
        )
        data["inference"].pop("generation")
        pipelines = build_pipelines(parse_config(data))
        pipelines.run_ingestion()
        result = pipelines.query("VPN 登入失敗")
        assert result["answer"] is None
        assert result["routing"]["category"] == "資訊系統"

    def test_custom_routing_via_file(self, tmp_path, corpus_dir):
        file = tmp_path / "router.py"
        file.write_text(ROUTER_OK, encoding="utf-8")
        config = routing_config(
            corpus_dir,
            {
                "method": "custom",
                "params": {"file": str(file), "class": "FixedRouter"},
            },
        )
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()
        result = pipelines.query("任何問題")
        assert result["routing"] == {"category": "測試類別", "source": "custom"}

    def test_custom_routing_contract_violation(self, tmp_path):
        file = tmp_path / "router.py"
        file.write_text(ROUTER_BAD_OUTPUT, encoding="utf-8")
        config = parse_config(
            make_config(
                inference={
                    "routing": {
                        "method": "custom",
                        "params": {"file": str(file), "class": "BadRouter"},
                    }
                }
            )
        )
        with pytest.raises(ConfigError, match="缺少輸出 socket") as excinfo:
            build_pipelines(config)
        assert "route" in str(excinfo.value)

    def test_routing_unknown_method_lists_alternatives(self):
        config = parse_config(
            make_config(inference={"routing": {"method": "llm_classify"}})
        )
        with pytest.raises(UnknownMethodError) as excinfo:
            build_pipelines(config)
        message = str(excinfo.value)
        assert "'keyword_match'" in message
        assert "'custom'" in message

    def test_routing_rejects_method_chain(self):
        config = parse_config(
            make_config(
                inference={
                    "routing": {
                        "method": ["keyword_match", "custom"],
                        "method_params": {
                            "keyword_match": {"routes": ROUTES},
                            "custom": {"class_path": "a:B"},
                        },
                    }
                }
            )
        )
        with pytest.raises(ConfigError, match="不支援方法鏈"):
            build_pipelines(config)

    def test_keyword_match_requires_routes(self):
        config = parse_config(
            make_config(
                inference={"routing": {"method": "keyword_match", "params": {}}}
            )
        )
        with pytest.raises(ConfigError, match="routes"):
            build_pipelines(config)


class TestTraceFormatting:
    def test_format_query_trace_includes_routing(self, corpus_dir):
        config = routing_config(corpus_dir, ROUTING_KEYWORD)
        pipelines = build_pipelines(config)
        pipelines.run_ingestion()
        result = pipelines.query("特休還剩幾天?")
        lines = format_query_trace(result["trace"])
        joined = "\n".join(lines)
        assert "Routing" in joined
        assert "人資/差勤" in joined

    def test_format_query_trace_without_routing_key(self):
        """舊版 trace dict(無 routing key)照常排版(.get 回歸保護)。"""
        trace = {
            "query": "q",
            "transforms": [],
            "subqueries": [],
            "fusion": {"documents": [], "applied": False},
            "generation": None,
        }
        lines = format_query_trace(trace)
        assert not any("Routing" in line for line in lines)


# --- 服務模式(fastapi 未安裝時 skip,與 test_service.py 同慣例)---

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from rag.service import create_app  # noqa: E402


def _service_config(corpus_dir, routing=None):
    data = copy.deepcopy(BASE_CONFIG)
    data["ingestion"]["import"]["params"]["input_dir"] = str(corpus_dir)
    data["inference"]["generation"] = {
        "method": "mock",
        "params": {"replies": ["mock 回答"]},
    }
    if routing is not None:
        data["inference"]["routing"] = routing
    return data


def _write_config(tmp_path, data):
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


class TestService:
    def test_query_response_includes_routing(self, tmp_path, corpus_dir):
        path = _write_config(
            tmp_path, _service_config(corpus_dir, routing=ROUTING_KEYWORD)
        )
        client = TestClient(create_app(path))
        body = client.post("/query", json={"query": "如何請假?"}).json()
        assert body["routing"]["category"] == "人資/差勤"

    def test_query_response_routing_null_when_unset(self, tmp_path, corpus_dir):
        path = _write_config(tmp_path, _service_config(corpus_dir))
        client = TestClient(create_app(path))
        body = client.post("/query", json={"query": "FAISS?"}).json()
        assert body["routing"] is None  # 向後相容:未設 routing 槽位

    def test_reload_keeps_routing_enabled(self, tmp_path, corpus_dir):
        """/reload 重建 RagPipelines 時 routing_enabled 必須跟著傳遞。"""
        path = _write_config(
            tmp_path, _service_config(corpus_dir, routing=ROUTING_KEYWORD)
        )
        client = TestClient(create_app(path))
        assert client.post("/reload").status_code == 200
        body = client.post("/query", json={"query": "薪資何時發放?"}).json()
        assert body["routing"] is not None
        assert body["routing"]["category"] == "人資/差勤"

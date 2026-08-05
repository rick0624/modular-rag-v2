"""服務模式測試:TestClient + 全離線 config(mock embedding + in_memory)。

需要 fastapi(``pip install -e ".[service]"``);未安裝時整檔 skip。
"""

from __future__ import annotations

import copy

import pytest
import yaml
from conftest import BASE_CONFIG

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from rag.config import load_raw_config  # noqa: E402
from rag.errors import ConfigError  # noqa: E402
from rag.kb_meta import ingestion_fingerprint  # noqa: E402
from rag.service import create_app, ingest_only  # noqa: E402


def _service_config(corpus_dir, **overrides):
    """離線服務 config:語料指向 tmp corpus、mock 生成給定回覆。"""
    data = copy.deepcopy(BASE_CONFIG)
    data["ingestion"]["import"]["params"]["input_dir"] = str(corpus_dir)
    data["inference"]["generation"] = {
        "method": "mock",
        "params": {"replies": ["這是 mock 回答。"]},
    }
    for path, value in overrides.items():
        section, key = path.split(".", 1)
        data[section][key] = value
    return data


def _write_config(tmp_path, data):
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def service(tmp_path, corpus_dir):
    """已啟動(含 ingestion)的 TestClient 與 config 路徑。"""
    path = _write_config(tmp_path, _service_config(corpus_dir))
    client = TestClient(create_app(path))
    return client, path, corpus_dir


def test_health_reports_fingerprint(service):
    client, _, _ = service
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexing_method"] == "in_memory"
    assert len(body["fingerprint"]) == 64  # sha256 hex


def test_query_returns_documents_and_answer(service):
    client, _, _ = service
    response = client.post("/query", json={"query": "FAISS 支援哪些索引?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "這是 mock 回答。"
    assert body["subquery_count"] == 1
    assert body["documents"], "啟動時已 ingest,應檢索得到內容"
    top = body["documents"][0]
    assert top["doc_id"] == "faiss.txt"
    assert top["chunk_id"].startswith("faiss.txt::chunk_")
    assert top["page"] == 1


def test_ingest_picks_up_new_files(service):
    client, _, corpus_dir = service
    (corpus_dir / "new.txt").write_text(
        "Qdrant 是向量資料庫,支援過濾與量化。", encoding="utf-8"
    )
    response = client.post("/ingest")
    assert response.status_code == 200
    body = response.json()
    assert body["documents_written"] > 0
    assert any(step["component"] == "writer" for step in body["steps"])

    hits = client.post("/query", json={"query": "Qdrant 向量資料庫"}).json()
    assert any(doc["doc_id"] == "new.txt" for doc in hits["documents"])


def test_reload_applies_inference_change(service, tmp_path):
    client, path, corpus_dir = service
    changed = _service_config(
        corpus_dir,
        **{
            "inference.generation": {
                "method": "mock",
                "params": {"replies": ["重載後的新回答。"]},
            }
        },
    )
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")

    response = client.post("/reload")
    assert response.status_code == 200
    assert response.json()["status"] == "reloaded"
    answer = client.post("/query", json={"query": "FAISS?"}).json()["answer"]
    assert answer == "重載後的新回答。"


def test_reload_rejects_ingestion_change_with_409(service):
    client, path, corpus_dir = service
    changed = _service_config(
        corpus_dir,
        **{
            "ingestion.chunking": {
                "method": "fixed_size",
                "params": {"split_length": 123, "split_overlap": 0},
            }
        },
    )
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")

    response = client.post("/reload")
    assert response.status_code == 409
    assert "/ingest" in response.json()["detail"]
    # 舊狀態不受影響,查詢照常
    assert client.post("/query", json={"query": "FAISS?"}).status_code == 200


def test_ingest_is_the_sanctioned_path_for_ingestion_change(service):
    """ingestion 變更走 /ingest 後,同一份 YAML 再 /reload 應通過。"""
    client, path, corpus_dir = service
    old_fingerprint = client.get("/health").json()["fingerprint"]
    changed = _service_config(
        corpus_dir,
        **{
            "ingestion.chunking": {
                "method": "fixed_size",
                "params": {"split_length": 123, "split_overlap": 0},
            }
        },
    )
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")

    ingest = client.post("/ingest")
    assert ingest.status_code == 200
    assert ingest.json()["fingerprint"] != old_fingerprint
    assert client.post("/reload").status_code == 200


def test_stage_inference_starts_with_empty_index(tmp_path, corpus_dir):
    path = _write_config(tmp_path, _service_config(corpus_dir))
    client = TestClient(create_app(path, stage="inference"))
    body = client.post("/query", json={"query": "FAISS?"}).json()
    assert body["documents"] == []  # in_memory + 跳過 ingest = 空索引,不炸


def test_stage_inference_still_allows_ingest_endpoint(tmp_path, corpus_dir):
    """啟動時沒建 ingestion pipeline,POST /ingest 仍是重建索引的正道。"""
    path = _write_config(tmp_path, _service_config(corpus_dir))
    client = TestClient(create_app(path, stage="inference"))
    assert client.post("/ingest").json()["documents_written"] > 0
    assert client.post("/query", json={"query": "FAISS?"}).json()["documents"]


class TestIngestOnly:
    """``serve.py --stage ingestion``:只建索引、不開服務。"""

    def test_builds_index_and_stamps_fingerprint(self, tmp_path, corpus_dir):
        path = _write_config(tmp_path, _service_config(corpus_dir))
        result = ingest_only(path)

        assert result["writer"]["documents_written"] > 0
        assert len(result["fingerprint"]) == 64
        # 指紋與 create_app 算出來的是同一個(兩個 process 之間的握手)
        assert result["fingerprint"] == ingestion_fingerprint(load_raw_config(path))

    def test_inference_pipeline_is_not_built(self, tmp_path, corpus_dir):
        """reranker 等 inference 側元件不該被載入 —— 用不存在的方法驗證。"""
        data = _service_config(corpus_dir)
        data["inference"]["reranking"] = {"method": "does_not_exist"}
        path = _write_config(tmp_path, data)
        # inference 有錯也不影響只建索引(建構期根本沒碰 inference 槽位)
        assert ingest_only(path)["writer"]["documents_written"] > 0

    def test_create_app_rejects_ingestion_stage_with_pointer(self, tmp_path, corpus_dir):
        path = _write_config(tmp_path, _service_config(corpus_dir))
        with pytest.raises(ConfigError, match="ingest_only"):
            create_app(path, stage="ingestion")


def test_reload_with_invalid_config_returns_400(service):
    client, path, corpus_dir = service
    data = _service_config(corpus_dir)
    data["inference"]["retrieval"] = {"method": "does_not_exist"}
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    response = client.post("/reload")
    assert response.status_code == 400
    assert "does_not_exist" in response.json()["detail"]


def test_query_response_includes_formatter_output(tmp_path, corpus_dir):
    data = _service_config(corpus_dir)
    data["inference"]["formatter"] = {"method": "simple_json"}
    path = _write_config(tmp_path, data)
    client = TestClient(create_app(path))

    body = client.post("/query", json={"query": "FAISS?"}).json()
    assert body["output"]["query"] == "FAISS?"
    assert body["output"]["total"] == len(body["documents"])

    # /reload 後 formatter 仍生效(formatter_enabled 有穿線)
    assert client.post("/reload").status_code == 200
    body = client.post("/query", json={"query": "BM25?"}).json()
    assert body["output"]["query"] == "BM25?"


def test_query_response_output_null_without_formatter(service):
    client, _, _ = service
    body = client.post("/query", json={"query": "FAISS?"}).json()
    assert body["output"] is None

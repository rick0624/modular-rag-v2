"""ingestion 指紋測試:穩定性、敏感範圍、儲存 roundtrip。"""

from __future__ import annotations

import copy

import yaml
from conftest import make_config

from haystack.document_stores.in_memory import InMemoryDocumentStore

from rag.kb_meta import ingestion_fingerprint, read_fingerprint, write_fingerprint


def test_stable_under_key_reordering_and_formatting():
    a = yaml.safe_load(
        "ingestion:\n"
        "  chunking: {method: fixed_size, params: {split_length: 300}}\n"
        "  embedding: {method: mock}\n"
        "inference: {retrieval: {method: bm25}}\n"
    )
    b = yaml.safe_load(
        "# 註解與排版不同、鍵順序顛倒 —— 語意相同\n"
        "inference:\n"
        "  retrieval:\n"
        "    method: bm25\n"
        "ingestion:\n"
        "  embedding:\n"
        "    method: mock\n"
        "  chunking:\n"
        "    params:\n"
        "      split_length: 300\n"
        "    method: fixed_size\n"
    )
    assert ingestion_fingerprint(a) == ingestion_fingerprint(b)


def test_stable_under_yaml_anchors():
    plain = yaml.safe_load("ingestion:\n  chunking: {split_length: 300}\n")
    anchored = yaml.safe_load(
        "defaults: &c {split_length: 300}\ningestion:\n  chunking: *c\n"
    )
    # anchor 展開後 ingestion 段相同;頂層多出的 defaults 鍵不在指紋範圍
    assert ingestion_fingerprint(plain) == ingestion_fingerprint(anchored)


def test_inference_and_evaluation_changes_ignored():
    base = make_config()
    changed = copy.deepcopy(base)
    changed["inference"]["retrieval"] = {"method": "hybrid", "params": {"top_k": 99}}
    changed["evaluation"] = {"method": "basic_retrieval_metrics", "params": {}}
    assert ingestion_fingerprint(base) == ingestion_fingerprint(changed)


def test_any_ingestion_subkey_change_detected():
    base = make_config()
    changed = copy.deepcopy(base)
    changed["ingestion"]["chunking"] = {
        "method": "fixed_size",
        "params": {"split_length": 123},
    }
    assert ingestion_fingerprint(base) != ingestion_fingerprint(changed)


def test_escape_hatch_path_change_detected():
    base = {"haystack_pipelines": {"ingestion": "a.yaml"}}
    changed = {"haystack_pipelines": {"ingestion": "b.yaml"}}
    assert ingestion_fingerprint(base) != ingestion_fingerprint(changed)


def test_env_var_values_never_enter_hash(monkeypatch):
    """指紋以展開前的原始 dict 計算:${SECRET} 的實際值變了也不影響。"""
    raw = make_config(
        ingestion={
            "embedding": {
                "method": "api_embedding",
                "params": {
                    "endpoint": "https://x/v1/embeddings",
                    "headers": {"Authorization": "Bearer ${SECRET_KEY}"},
                },
            }
        }
    )
    monkeypatch.setenv("SECRET_KEY", "value-one")
    first = ingestion_fingerprint(raw)
    monkeypatch.setenv("SECRET_KEY", "value-two")
    assert ingestion_fingerprint(raw) == first
    assert "value-one" not in first  # 雜湊輸入不含展開值(防呆斷言)


def test_incremental_flag_does_not_change_fingerprint():
    """incremental 是操作旗標,不影響索引內容,開關它不該觸發重建要求。"""
    base = make_config()
    flagged = copy.deepcopy(base)
    flagged["ingestion"]["indexing"] = {
        "method": "in_memory",
        "params": {"incremental": True},
    }
    plain = copy.deepcopy(base)
    plain["ingestion"]["indexing"] = {"method": "in_memory", "params": {}}
    assert ingestion_fingerprint(flagged) == ingestion_fingerprint(plain)

    # method_params 寫法也一樣
    flagged_mp = copy.deepcopy(base)
    flagged_mp["ingestion"]["indexing"] = {
        "method": "in_memory",
        "method_params": {"in_memory": {"incremental": True}},
    }
    plain_mp = copy.deepcopy(base)
    plain_mp["ingestion"]["indexing"] = {
        "method": "in_memory",
        "method_params": {"in_memory": {}},
    }
    assert ingestion_fingerprint(flagged_mp) == ingestion_fingerprint(plain_mp)


def test_in_memory_roundtrip():
    store = InMemoryDocumentStore()
    assert read_fingerprint("in_memory", store) is None
    write_fingerprint("in_memory", store, "abc123")
    assert read_fingerprint("in_memory", store) == "abc123"

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


def test_es_credentials_do_not_change_fingerprint():
    """認證欄位只影響連線,不影響索引內容:補上憑證不該要求重建索引。"""
    base = make_config()
    plain = copy.deepcopy(base)
    plain["ingestion"]["indexing"] = {
        "method": "elasticsearch",
        "params": {"hosts": "${ES_URL}", "index": "kb"},
    }
    authed = copy.deepcopy(base)
    authed["ingestion"]["indexing"] = {
        "method": "elasticsearch",
        "params": {
            "hosts": "${ES_URL}",
            "index": "kb",
            "username": "${ES_USERNAME}",
            "password": "${ES_PASSWORD}",
            "ca_certs": "/etc/ssl/company-ca.crt",
            "verify_certs": True,
        },
    }
    assert ingestion_fingerprint(authed) == ingestion_fingerprint(plain)


def test_custom_mapping_changes_fingerprint():
    """custom_mapping 決定索引怎麼建(analyzer / 欄位),改了 = 索引過期。"""
    base = make_config()
    plain = copy.deepcopy(base)
    plain["ingestion"]["indexing"] = {
        "method": "elasticsearch",
        "params": {"hosts": "${ES_URL}", "index": "kb"},
    }
    mapped = copy.deepcopy(plain)
    mapped["ingestion"]["indexing"]["params"]["custom_mapping"] = {
        "properties": {"content": {"type": "text", "analyzer": "ik_smart"}}
    }
    assert ingestion_fingerprint(mapped) != ingestion_fingerprint(plain)

    # 但索引位置本身變了,仍要被偵測到
    moved = copy.deepcopy(plain)
    moved["ingestion"]["indexing"]["params"]["index"] = "kb2"
    assert ingestion_fingerprint(moved) != ingestion_fingerprint(plain)


def test_in_memory_roundtrip():
    store = InMemoryDocumentStore()
    assert read_fingerprint("in_memory", store) is None
    write_fingerprint("in_memory", store, "abc123")
    assert read_fingerprint("in_memory", store) == "abc123"


# ---------------------------------------------------------------------------
# ingestion custom module 的 file: 內容進指紋
# ---------------------------------------------------------------------------


def _custom_parsing_config(file_path: str) -> dict:
    return make_config(
        ingestion={
            "parsing": {
                "method": "custom",
                "params": {"file": file_path, "class": "P", "kind": "converter"},
            }
        }
    )


def test_custom_module_content_change_changes_fingerprint(tmp_path):
    module = tmp_path / "parser.py"
    module.write_text("VERSION = 1\n", encoding="utf-8")
    config = _custom_parsing_config(str(module))
    before = ingestion_fingerprint(config)

    module.write_text("VERSION = 2\n", encoding="utf-8")
    assert ingestion_fingerprint(config) != before  # 路徑沒變,內容變了也要偵測


def test_fingerprint_without_custom_is_byte_identical_to_legacy():
    """沒用 ingestion custom 的 config,指紋公式必須與加入本機制前完全相同。

    以「舊公式」(不含 custom_module_files 鍵)行內重算對照 —— 換了公式,
    既有 ES 索引上存的指紋會全部對不上,升級即全面誤報「需重建」。
    """
    import hashlib
    import json

    from rag.kb_meta import _strip_operational_keys

    raw = make_config()
    legacy_payload = {
        "ingestion": _strip_operational_keys(raw.get("ingestion")),
        "haystack_pipelines.ingestion": (
            (raw.get("haystack_pipelines") or {}).get("ingestion")
        ),
    }
    legacy = hashlib.sha256(
        json.dumps(
            legacy_payload, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), default=str,
        ).encode("utf-8")
    ).hexdigest()
    assert ingestion_fingerprint(raw) == legacy


def test_unreadable_custom_file_falls_back_to_path_only(tmp_path):
    config = _custom_parsing_config(str(tmp_path / "no_such.py"))
    fingerprint = ingestion_fingerprint(config)  # 不炸
    # 讀不到 → 內容不進指紋,但路徑仍在:換路徑仍會偵測到
    other = _custom_parsing_config(str(tmp_path / "other_missing.py"))
    assert ingestion_fingerprint(other) != fingerprint


def test_class_path_not_content_hashed():
    """class_path 指向已安裝套件,無單一檔案可雜湊 —— 只有路徑字串進指紋。"""
    config = make_config(
        ingestion={
            "parsing": {
                "method": "custom",
                "params": {"class_path": "my_pkg.parsers:P", "kind": "converter"},
            }
        }
    )
    from rag.kb_meta import custom_file_hashes

    assert custom_file_hashes(config["ingestion"]) == {}


def test_inference_side_custom_file_not_hashed(tmp_path):
    """inference 端的 custom 改了走 /reload 即生效,不進 ingestion 指紋。"""
    module = tmp_path / "reranker.py"
    module.write_text("VERSION = 1\n", encoding="utf-8")
    config = make_config(
        inference={
            "reranking": {
                "method": "custom",
                "params": {"file": str(module), "class": "R"},
            }
        }
    )
    before = ingestion_fingerprint(config)
    module.write_text("VERSION = 2\n", encoding="utf-8")
    assert ingestion_fingerprint(config) == before


def test_method_params_custom_block_wins_for_hashing(tmp_path):
    """file 的取用優先序與 params_for 一致:method_params.custom 蓋過 params。"""
    used = tmp_path / "used.py"
    used.write_text("A = 1\n", encoding="utf-8")
    config = make_config(
        ingestion={
            "chunking": {
                "method": "custom",
                "params": {"file": str(tmp_path / "ignored.py"), "class": "X"},
                "method_params": {"custom": {"file": str(used), "class": "X"}},
            }
        }
    )
    before = ingestion_fingerprint(config)
    used.write_text("A = 2\n", encoding="utf-8")
    assert ingestion_fingerprint(config) != before

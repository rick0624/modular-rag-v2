"""experiment.py 組合生成測試:bundle 綁定、維度重疊防呆、label 命名。

只測純函式(make_variants / resolve_overrides),不跑管線。
scripts/ 不是 package,以 importlib 從檔案路徑載入模組。
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "experiment.py"


@pytest.fixture
def experiment():
    """載入 experiment 模組,結束後還原它 import 時關掉的 logging。"""
    spec = importlib.util.spec_from_file_location("experiment", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    logging.disable(logging.NOTSET)  # 模組層有 logging.disable(WARNING)


# 最小 base config:make_variants 只碰 section/slot 兩層,不需要真 schema。
def make_base() -> dict:
    return {
        "ingestion": {"embedding": {"method": "mock"}},
        "inference": {
            "retrieval": {"method": "hybrid"},
            "query_transformation": {"method": "passthrough"},
        },
    }


BUNDLE_A = {
    "_label": "stack-a",
    "ingestion.embedding": {"method": "sentence_transformers"},
    "inference.retrieval": {"method": "custom", "params": {"file": "a.py"}},
}
BUNDLE_B = {
    "_label": "stack-b",
    "ingestion.embedding": {"method": "mock"},
    "inference.retrieval": {"method": "embedding"},
}


def test_product_bundle_moves_slots_together(experiment, monkeypatch):
    monkeypatch.setattr(experiment, "MODE", "product")
    monkeypatch.setattr(experiment, "SLOT_OPTIONS", {
        "embedding_stack": [BUNDLE_A, BUNDLE_B],
        "inference.query_transformation": ["passthrough", "normalize"],
    })
    variants = experiment.make_variants(make_base())
    assert len(variants) == 4  # 2 bundle × 2 transformation,bundle 內不交叉
    for label, overrides, cfg in variants:
        # bundle 的兩個槽位永遠成對出現在同一組合
        emb = cfg["ingestion"]["embedding"]["method"]
        ret = cfg["inference"]["retrieval"]["method"]
        assert (emb, ret) in {("sentence_transformers", "custom"), ("mock", "embedding")}
        assert "_label" not in str(cfg)  # _label 不落入 config
        assert set(overrides) == {
            "ingestion.embedding", "inference.retrieval",
            "inference.query_transformation",
        }
    labels = [label for label, _, _ in variants]
    assert "embedding_stack=stack-a + query_transformation=passthrough" in labels
    assert "embedding_stack=stack-b + query_transformation=normalize" in labels


def test_one_at_a_time_applies_bundle_whole(experiment, monkeypatch):
    monkeypatch.setattr(experiment, "MODE", "one_at_a_time")
    monkeypatch.setattr(experiment, "SLOT_OPTIONS", {
        "embedding_stack": [BUNDLE_A],
        "inference.query_transformation": ["normalize"],
    })
    variants = experiment.make_variants(make_base())
    assert [label for label, _, _ in variants] == [
        "baseline", "embedding_stack=stack-a", "query_transformation=normalize",
    ]
    baseline, bundled, transformed = (cfg for _, _, cfg in variants)
    # bundle 整組套用
    assert bundled["ingestion"]["embedding"]["method"] == "sentence_transformers"
    assert bundled["inference"]["retrieval"]["method"] == "custom"
    # 其他維度輪動時 bundle 槽位保持 baseline
    assert transformed["ingestion"]["embedding"] == baseline["ingestion"]["embedding"]
    assert transformed["inference"]["retrieval"] == baseline["inference"]["retrieval"]


def test_bundle_without_label_gets_fallback_name(experiment, monkeypatch):
    bundle = {k: v for k, v in BUNDLE_A.items() if k != "_label"}
    monkeypatch.setattr(experiment, "MODE", "one_at_a_time")
    monkeypatch.setattr(experiment, "SLOT_OPTIONS", {"stack": [bundle]})
    labels = [label for label, _, _ in experiment.make_variants(make_base())]
    assert labels == ["baseline", "stack=embedding:sentence_transformers+retrieval:custom"]


def test_overlapping_dimensions_rejected(experiment, monkeypatch):
    monkeypatch.setattr(experiment, "SLOT_OPTIONS", {
        "embedding_stack": [BUNDLE_A],
        "inference.retrieval": ["bm25"],  # bundle 已覆蓋此槽位
    })
    with pytest.raises(ValueError, match="都覆蓋槽位 'inference.retrieval'"):
        experiment.make_variants(make_base())


def test_non_slot_dimension_requires_bundle_options(experiment, monkeypatch):
    monkeypatch.setattr(experiment, "SLOT_OPTIONS", {"stack": ["bm25"]})
    with pytest.raises(ValueError, match="不是槽位路徑"):
        experiment.make_variants(make_base())

"""配置層測試:schema 驗證、${ENV_VAR} 注入、.env 載入、escape hatch 互斥。"""

from __future__ import annotations

import copy

import pytest

from rag.config import MethodConfig, load_config, parse_config
from rag.errors import ConfigError

MINIMAL = {
    "ingestion": {
        "import": {"method": "local_file", "params": {"input_dir": "./data/raw"}},
        "parsing": {"method": "plain_text"},
        "chunking": {"method": "fixed_size"},
        "embedding": {"method": "mock"},
        "indexing": {"method": "in_memory"},
    },
    "inference": {
        "query_transformation": {"method": "passthrough"},
        "retrieval": {"method": "bm25"},
        "reranking": {"method": "llm"},
        "generation": {"method": "mock"},
    },
}


def make_config(**overrides):
    data = copy.deepcopy(MINIMAL)
    data.update(overrides)
    return data


class TestSchema:
    def test_minimal_config_parses(self):
        config = parse_config(make_config())
        assert config.ingestion.import_.method == "local_file"
        assert config.inference.retrieval.method == "bm25"
        assert config.evaluation is None

    def test_top_level_must_be_mapping(self):
        with pytest.raises(ConfigError, match="頂層必須是 mapping"):
            parse_config(["not", "a", "dict"])

    def test_extra_field_rejected(self):
        data = make_config()
        data["ingestion"]["chunking"]["chunk_size"] = 512  # 參數打錯位置
        with pytest.raises(ConfigError, match="驗證失敗"):
            parse_config(data)

    def test_missing_module_reports_location(self):
        data = make_config()
        del data["inference"]["retrieval"]
        with pytest.raises(ConfigError, match="inference.retrieval"):
            parse_config(data)

    def test_fusion_defaults_and_literals(self):
        data = make_config()
        data["inference"]["fusion"] = {"group_by": "doc"}
        config = parse_config(data)
        assert config.inference.fusion.strategy == "rrf"
        assert config.inference.fusion.top_k == 5

        data["inference"]["fusion"] = {"group_by": "paragraph"}
        with pytest.raises(ConfigError, match="fusion.group_by"):
            parse_config(data)

    def test_missing_config_file(self, tmp_path):
        with pytest.raises(ConfigError, match="找不到設定檔"):
            load_config(tmp_path / "nope.yaml", dotenv_path=None)


class TestFusionCustomShape:
    """fusion 的兩種寫法:內建扁平參數 × custom module,互斥。"""

    @staticmethod
    def _with_fusion(fusion):
        data = make_config()
        data["inference"]["fusion"] = fusion
        return data

    def test_custom_shape_parses(self):
        config = parse_config(
            self._with_fusion(
                {"method": "custom", "params": {"file": "./f.py", "class": "F"}}
            )
        )
        assert config.inference.fusion.method == "custom"
        assert config.inference.fusion.params["class"] == "F"

    def test_flat_shape_unchanged(self):
        config = parse_config(self._with_fusion({"strategy": "max_score"}))
        assert config.inference.fusion.method is None
        assert config.inference.fusion.strategy == "max_score"

    def test_mixing_custom_with_builtin_knobs_rejected(self):
        with pytest.raises(ConfigError, match="請移除內建參數"):
            parse_config(
                self._with_fusion(
                    {"method": "custom", "params": {"file": "./f.py"}, "top_k": 5}
                )
            )

    def test_unknown_method_rejected(self):
        with pytest.raises(ConfigError, match="只支援 'custom'"):
            parse_config(self._with_fusion({"method": "rrf"}))

    def test_params_without_method_rejected(self):
        with pytest.raises(ConfigError, match="只搭配 method: custom"):
            parse_config(self._with_fusion({"params": {"file": "./f.py"}}))


class TestMethodParams:
    def test_method_params_block_wins_over_params(self):
        cfg = MethodConfig(
            method="a",
            params={"x": 1},
            method_params={"a": {"y": 2}, "b": {"z": 3}},
        )
        assert cfg.params_for() == {"y": 2}

    def test_params_used_when_no_block(self):
        cfg = MethodConfig(method="a", params={"x": 1}, method_params={"b": {"z": 3}})
        assert cfg.params_for() == {"x": 1}

    def test_params_for_specific_method(self):
        cfg = MethodConfig(method=["a", "b"], method_params={"a": {"x": 1}, "b": {"y": 2}})
        assert cfg.params_for("b") == {"y": 2}

    def test_chain_with_flat_params_rejected(self):
        data = make_config()
        data["inference"]["reranking"] = {
            "method": ["similarity", "llm"],
            "params": {"top_k": 5},
        }
        with pytest.raises(ConfigError, match="method_params"):
            parse_config(data)

    def test_empty_chain_rejected(self):
        with pytest.raises(ConfigError, match="method 清單不可為空"):
            parse_config(
                make_config(
                    inference={
                        **MINIMAL["inference"],
                        "reranking": {"method": []},
                    }
                )
            )

    def test_methods_normalizes_to_list(self):
        assert MethodConfig(method="a").methods() == ["a"]
        assert MethodConfig(method=["a", "b"], method_params={}).methods() == ["a", "b"]


class TestEnvVarInjection:
    def test_expansion(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "secret-123")
        data = make_config()
        data["inference"]["generation"] = {
            "method": "gateway_openai_compatible",
            "params": {"base_url": "https://x", "api_key": "${TEST_API_KEY}"},
        }
        config = parse_config(data)
        assert config.inference.generation.params["api_key"] == "secret-123"

    def test_missing_env_var_fails_loud(self, monkeypatch):
        monkeypatch.delenv("SURELY_NOT_SET_VAR", raising=False)
        data = make_config()
        data["inference"]["generation"]["params"] = {"api_key": "${SURELY_NOT_SET_VAR}"}
        with pytest.raises(ConfigError, match="未設定的環境變數 'SURELY_NOT_SET_VAR'"):
            parse_config(data)

    def test_escape_produces_literal(self):
        data = make_config()
        data["inference"]["generation"]["params"] = {"template": "字面值 $${NOT_A_VAR}"}
        config = parse_config(data)
        assert config.inference.generation.params["template"] == "字面值 ${NOT_A_VAR}"


class TestDotenv:
    def _write_yaml(self, tmp_path, content: str):
        path = tmp_path / "config.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    def _minimal_yaml_with_key(self) -> str:
        return """
ingestion:
  import: {method: local_file}
  parsing: {method: plain_text}
  chunking: {method: fixed_size}
  embedding: {method: mock}
  indexing: {method: in_memory}
inference:
  query_transformation: {method: passthrough}
  retrieval: {method: bm25}
  reranking: {method: llm}
  generation:
    method: mock
    params:
      api_key: ${FROM_DOTENV_KEY}
"""

    def test_dotenv_autoload(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FROM_DOTENV_KEY", raising=False)
        (tmp_path / ".env").write_text("FROM_DOTENV_KEY=dotenv-value\n", encoding="utf-8")
        config_path = self._write_yaml(tmp_path, self._minimal_yaml_with_key())
        config = load_config(config_path, dotenv_path=tmp_path / ".env")
        assert config.inference.generation.params["api_key"] == "dotenv-value"

    def test_real_env_wins_over_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROM_DOTENV_KEY", "real-env-value")
        (tmp_path / ".env").write_text("FROM_DOTENV_KEY=dotenv-value\n", encoding="utf-8")
        config_path = self._write_yaml(tmp_path, self._minimal_yaml_with_key())
        config = load_config(config_path, dotenv_path=tmp_path / ".env")
        assert config.inference.generation.params["api_key"] == "real-env-value"

    def test_bad_dotenv_line_reports_line_number(self, tmp_path):
        (tmp_path / ".env").write_text("OK=1\nthis is not valid\n", encoding="utf-8")
        config_path = self._write_yaml(tmp_path, self._minimal_yaml_with_key())
        with pytest.raises(ConfigError, match="第 2 行"):
            load_config(config_path, dotenv_path=tmp_path / ".env")


class TestRequiredStages:
    def test_both_stages_must_be_present(self):
        data = make_config()
        del data["inference"]
        with pytest.raises(ConfigError, match="缺少 'inference' 配置"):
            parse_config(data)

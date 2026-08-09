"""YAML 配置的 schema 定義與載入(含 ${ENV_VAR} 展開與 .env 載入)。

整條 pipeline 由單一 YAML config 控制:每個槽位指定 ``method`` 與該
方法專屬的 ``params``。這裡只驗證「結構層」(每個槽位都要有 method /
params);``params`` 內容的驗證交給各方法型錄的 factory,
因此新增方法時完全不需改動本檔案。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rag.errors import ConfigError

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ESCAPE_SENTINEL = "\x00ESCAPED_DOLLAR_BRACE\x00"

# --- .env 載入(不依賴 python-dotenv)--------------------------------------

_DOTENV_LINE_PATTERN = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _parse_dotenv(text: str, source: str = ".env") -> dict[str, str]:
    """把 .env 內容解析成 dict(不寫入環境變數)。

    支援 ``KEY=value``、``export`` 前綴、單/雙引號包裹的值、
    未加引號值的行內註解(以「空白 + #」分隔)。

    Raises:
        ConfigError: 任一行不是 ``KEY=value`` 格式。
    """
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _DOTENV_LINE_PATTERN.match(line)
        if match is None:
            raise ConfigError(
                f"'{source}' 第 {line_number} 行格式不正確:{raw_line!r}"
                "(應為 KEY=value)"
            )
        key, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            # 未加引號的值:移除行內註解(以「空白 + #」分隔)。
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """讀取 .env 檔並寫入 ``os.environ``;檔案不存在時安靜跳過。

    預設不覆蓋已存在的環境變數(export / 系統設定優先於 .env)。

    Returns:
        實際寫入環境的變數 dict(被跳過的不含在內)。

    Raises:
        ConfigError: 檔案存在但內容格式不正確。
    """
    env_path = Path(path)
    if not env_path.is_file():
        return {}
    values = _parse_dotenv(env_path.read_text(encoding="utf-8"), source=str(env_path))
    applied: dict[str, str] = {}
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def _expand_env_in_str(text: str, source: str) -> str:
    """展開單一字串中的 ``${ENV_VAR}`` 佔位符。"""
    escaped = text.replace("$${", _ESCAPE_SENTINEL)

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = os.environ.get(name)
        if value is None:
            raise ConfigError(
                f"配置 '{source}' 引用了未設定的環境變數 '{name}'"
                f"(於字串 {text!r})。請先設定該環境變數;"
                "若要輸出字面值 ${...},請寫成 $${...}"
            )
        return value

    return _ENV_VAR_PATTERN.sub(_replace, escaped).replace(_ESCAPE_SENTINEL, "${")


def expand_env_vars(value: Any, source: str = "<dict>") -> Any:
    """遞迴展開設定值中所有字串的 ``${ENV_VAR}`` 佔位符。

    機密(API key 等)因此不必寫死在 YAML 中,改由環境變數注入,
    設定檔可以安心進版控。``$${...}`` 會輸出字面值 ``${...}``。

    Args:
        value: YAML 解析後的任意節點(str / dict / list 會被遞迴處理)。
        source: 資料來源描述,用於錯誤訊息。

    Raises:
        ConfigError: 引用的環境變數未設定。
    """
    if isinstance(value, str):
        return _expand_env_in_str(value, source)
    if isinstance(value, dict):
        return {key: expand_env_vars(item, source) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item, source) for item in value]
    return value


class MethodConfig(BaseModel):
    """單一槽位的配置:方法名稱 + 參數。

    ``method`` 也可以寫成**清單**(方法鏈):輸入輸出同型別的槽位
    (parsing、query_transformation、reranking)會依序執行清單中的方法,
    用於疊加處理(例如 ``[pdf, clean]``、``[similarity, llm]``)。
    其他槽位不支援清單,builder 會在建構期報錯。

    參數有兩種寫法,可以並用:

    1. ``params``:扁平寫法,直接放「當前 method」的參數(單一方法時最簡潔)。
    2. ``method_params``:以方法名稱分區,每個方法的參數放在自己的區塊內。
       多個方法的設定可同時保留在 config 中,切換 ``method`` 時只會取用
       對應區塊,其餘區塊不會被驗證、也不會互相衝突。

    有效參數的決定規則(見 :meth:`params_for`):``method_params`` 中存在
    當前方法的區塊時,採用該區塊並忽略 ``params``;否則使用 ``params``。
    方法鏈(清單長度 > 1)時 ``params`` 會產生「屬於哪個方法」的歧義,
    因此必須為空,一律改用 ``method_params`` 分區。
    """

    model_config = ConfigDict(extra="forbid")

    method: str | list[str] = Field(
        description="方法名稱;同型槽位可寫清單(方法鏈,依序執行)"
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="當前 method 的參數(扁平寫法;方法鏈時必須為空)",
    )
    method_params: dict[str, dict[str, Any] | None] = Field(
        default_factory=dict,
        description="各方法專屬參數,以方法名稱分區;切換 method 時互不干擾",
    )

    @field_validator("method_params", mode="after")
    @classmethod
    def _null_block_is_empty(
        cls, value: dict[str, dict[str, Any] | None]
    ) -> dict[str, dict[str, Any]]:
        """把 ``None`` 區塊視為空參數。

        參數全被註解掉的區塊在 YAML 中是 ``null`` 而非 ``{}``
        (型錄式 config 很常見),不該因此驗證失敗。
        """
        return {key: (block or {}) for key, block in value.items()}

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list):
            if not value:
                raise ValueError("method 清單不可為空")
            if not all(isinstance(item, str) and item for item in value):
                raise ValueError("method 清單的元素必須是非空字串")
        return value

    @model_validator(mode="after")
    def _reject_ambiguous_params(self) -> "MethodConfig":
        if isinstance(self.method, list) and len(self.method) > 1 and self.params:
            raise ValueError(
                "method 為方法鏈(多個方法)時,params 無法判斷屬於哪個方法;"
                "請改用 method_params 以方法名稱分區"
            )
        return self

    def methods(self) -> list[str]:
        """回傳方法名稱清單(單一方法回傳長度 1 的清單)。"""
        return list(self.method) if isinstance(self.method, list) else [self.method]

    def params_for(self, method: str | None = None) -> dict[str, Any]:
        """回傳指定方法(預設為第一個 ``method``)的有效參數。

        ``method_params`` 中有該方法的區塊時,回傳該區塊(忽略 ``params``,
        避免其他方法的參數混入);否則回傳 ``params``。
        """
        target = method or self.methods()[0]
        if target in self.method_params:
            return dict(self.method_params[target])
        return dict(self.params)


class IngestionConfig(BaseModel):
    """Ingestion 階段(import → parsing → chunking → embedding → indexing)的配置。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    import_: MethodConfig = Field(alias="import", description="槽位:Import")
    parsing: MethodConfig = Field(description="槽位:Parsing(可方法鏈)")
    chunking: MethodConfig = Field(description="槽位:Chunking")
    embedding: MethodConfig = Field(description="槽位:Embedding(inference 端查詢向量同源派生)")
    indexing: MethodConfig = Field(description="槽位:Indexing(document store)")


class FusionConfig(BaseModel):
    """融合 / 聚合步驟的設定。

    兩種寫法,擇一:

    - **內建**(原有扁平寫法):``group_by`` / ``strategy`` / ``top_k``。
      query transformation 拆出多個子查詢、或有此區塊時生效;
      都沒有時此步驟等於不存在(單一查詢原樣通過)。
    - **custom module**:``method: custom`` + ``params``(file/class 或
      class_path,init_params 透傳)。掛上即**一律執行**(單一查詢也進
      元件,原樣通過與否由元件自己決定),因此不可與內建參數混用。
    """

    model_config = ConfigDict(extra="forbid")

    method: str | None = Field(
        default=None,
        description="'custom' 掛自訂融合元件;省略(None)= 內建融合",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="method: custom 的參數(file/class 或 class_path、init_params)",
    )
    group_by: Literal["none", "doc", "page"] = Field(
        default="none", description="聚合粒度:切片 / 文件 / 頁(僅內建)"
    )
    strategy: Literal["rrf", "concat_dedup", "max_score"] = Field(
        default="rrf", description="排序策略(rrf 為名次法,跨子查詢最安全;僅內建)"
    )
    top_k: int = Field(default=5, gt=0, description="融合後保留的筆數上限(僅內建)")

    @model_validator(mode="after")
    def _builtin_xor_custom(self) -> "FusionConfig":
        # model_fields_set 只含 YAML 顯式寫出的欄位:顯式寫預設值也算混用
        # (使用者以為參數會生效,實際被 custom 元件忽略 —— 必須擋)。
        flat_given = {"group_by", "strategy", "top_k"} & self.model_fields_set
        if self.method is None:
            if self.params:
                raise ValueError(
                    "fusion.params 只搭配 method: custom 使用;"
                    "內建融合直接寫 group_by / strategy / top_k"
                )
            return self
        if self.method != "custom":
            raise ValueError(
                f"fusion 的 method 目前只支援 'custom'(收到 '{self.method}');"
                "內建融合請省略 method,直接寫 group_by / strategy / top_k"
            )
        if flat_given:
            listed = ", ".join(sorted(flat_given))
            raise ValueError(
                f"fusion 同時設定了 method: custom 與內建參數({listed});"
                "custom 元件的行為由 params.init_params 決定,請移除內建參數"
            )
        return self


class InferenceConfig(BaseModel):
    """Inference 階段(query_transformation → retrieval → reranking → generation)的配置。"""

    model_config = ConfigDict(extra="forbid")

    query_transformation: MethodConfig = Field(description="槽位:Query Transformation(可方法鏈)")
    retrieval: MethodConfig = Field(description="槽位:Retrieval")
    reranking: MethodConfig = Field(description="槽位:Reranking(可方法鏈)")
    generation: MethodConfig | None = Field(
        default=None,
        description="槽位:Generation(generate_answer: false 時可省略;"
        "保留時作為各 LLM 方法沿用的預設連線)",
    )
    generate_answer: bool = Field(
        default=True,
        description="是否執行答案生成;false 時 pipeline 止於 fusion(檢索-only)",
    )
    fusion: FusionConfig | None = Field(
        default=None,
        description="融合 / 聚合步驟(選填;多子查詢或 doc/page 聚合時使用)",
    )
    routing: MethodConfig | None = Field(
        default=None,
        description="槽位:Question Routing(選填;判斷查詢類別,結果附加於"
        "輸出的 routing key,不影響檢索行為)",
    )
    formatter: MethodConfig | None = Field(
        default=None,
        description="槽位:Formatter(選填;fusion 之後的終端支線,把最終"
        "結果組成對外格式,結果進 query() 輸出的 output key;"
        "省略時 output 為 None)",
    )

    @model_validator(mode="after")
    def _generation_required_unless_skipped(self) -> "InferenceConfig":
        if self.generate_answer and self.generation is None:
            raise ValueError(
                "generate_answer 為 true(預設)時必須提供 generation 槽位;"
                "檢索-only 模式請設定 generate_answer: false"
            )
        return self


class RAGConfig(BaseModel):
    """整條 pipeline 的頂層配置(ingestion 與 inference 皆必填)。"""

    model_config = ConfigDict(extra="forbid")

    ingestion: IngestionConfig | None = Field(
        default=None, description="Ingestion 槽位配置"
    )
    inference: InferenceConfig | None = Field(
        default=None, description="Inference 槽位配置"
    )
    evaluation: MethodConfig | None = Field(
        default=None, description="Evaluation 配置(可省略)"
    )

    @model_validator(mode="after")
    def _both_stages_present(self) -> "RAGConfig":
        for stage in ("ingestion", "inference"):
            if getattr(self, stage) is None:
                raise ValueError(
                    f"缺少 '{stage}' 配置:請提供 '{stage}' 槽位區塊"
                )
        return self


def _format_validation_error(exc: ValidationError, source: str) -> str:
    """把 pydantic 的 ValidationError 轉成人類可讀的多行訊息。"""
    lines = [f"配置 '{source}' 驗證失敗,共 {exc.error_count()} 個錯誤:"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        raw_input = repr(err.get("input"))
        if len(raw_input) > 80:
            raw_input = raw_input[:77] + "..."
        lines.append(f"  - {loc}: {err['msg']}(輸入值:{raw_input})")
    return "\n".join(lines)


def parse_config(data: Any, source: str = "<dict>") -> RAGConfig:
    """把已解析的 dict 驗證成 :class:`RAGConfig`。

    驗證前會先以 :func:`expand_env_vars` 展開字串值中的 ``${ENV_VAR}``。

    Args:
        data: YAML 解析後的物件(必須是 mapping)。
        source: 資料來源描述,用於錯誤訊息。

    Raises:
        ConfigError: 結構不是 mapping、環境變數未設定,
            或欄位缺漏 / 型別錯誤。
    """
    if not isinstance(data, dict):
        raise ConfigError(
            f"配置 '{source}' 的頂層必須是 mapping(YAML 物件),"
            f"實際得到:{type(data).__name__}"
        )
    data = expand_env_vars(data, source)
    try:
        return RAGConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, source)) from exc


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """讀取 YAML 為原始 dict:不載入 .env、不展開 ``${ENV_VAR}``、不驗證。

    供 ingestion 指紋(:mod:`rag.kb_meta`)使用 —— 指紋必須以「展開前」
    的內容計算,機密才不會進雜湊;也因為回傳的是解析後的 dict,
    註解、排版與 YAML anchor 的重構都不影響指紋。

    Raises:
        ConfigError: 檔案不存在、YAML 語法錯誤,或頂層不是 mapping。
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"找不到設定檔:{config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"設定檔 '{config_path}' 不是合法的 YAML:{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"配置 '{config_path}' 的頂層必須是 mapping(YAML 物件),"
            f"實際得到:{type(data).__name__}"
        )
    return data


def load_config(
    path: str | Path, dotenv_path: str | Path | None = ".env"
) -> RAGConfig:
    """從 YAML 檔載入並驗證整條 pipeline 的配置。

    載入前會先讀取 ``dotenv_path`` 指定的 .env 檔(預設為目前工作目錄
    下的 ``.env``,不存在時安靜跳過),再展開 config 中的 ``${ENV_VAR}``。
    已存在的環境變數不會被 .env 覆蓋。

    Args:
        path: YAML 設定檔路徑。
        dotenv_path: .env 檔路徑;傳 ``None`` 可停用自動載入。

    Raises:
        ConfigError: 檔案不存在、YAML 語法錯誤、.env 格式錯誤,
            或 schema 驗證失敗。
    """
    if dotenv_path is not None:
        load_dotenv(dotenv_path)
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"找不到設定檔:{config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"設定檔 '{config_path}' 不是合法的 YAML:{exc}") from exc
    return parse_config(data, source=str(config_path))

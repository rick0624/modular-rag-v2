"""槽位系統的共用底座:SlotFactory / BuildContext / 參數驗證 / 圖片段。

方法型錄(:mod:`rag.methods_ingestion` / :mod:`rag.methods_inference`)與
組裝層(:mod:`rag.builder`)共用的最小基礎;evaluation 等其他模組也從
這裡取用 :class:`BaseParams` 與 :func:`validate_params`,不必依賴 builder。
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from rag.errors import ConfigError, UnknownMethodError

if TYPE_CHECKING:
    from rag.config import MethodConfig

# 輸入輸出同型別、支援方法鏈(method 清單)的槽位。
CHAINABLE_SLOTS = frozenset({"parsing", "query_transformation", "reranking"})


@dataclass(frozen=True)
class SlotFactory:
    """一個方法的建立函式 + 宣告式相容性欄位。

    ``build(params, ctx)`` 回傳該方法對應的 Haystack 元件
    (embedding 槽位回傳 ``(document_embedder, text_embedder)`` 一對,
    indexing 槽位回傳 document store,``no_chunking`` 等可回傳 None,
    retrieval / auto parsing 回傳 :class:`SlotGraph`)。

    custom 方法的相容性宣告(kind / produces_pages / requires_pages…)寫在
    config 參數裡,不在 factory 上 —— 由 :mod:`rag.methods_ingestion` 的
    ``parsing_declaration`` / ``chunking_requires_pages`` 於建構期讀出。
    """

    build: Callable[[dict[str, Any], "BuildContext"], Any]
    kind: str = ""  # parsing 鏈用:"converter"(檔案→Document)/ "doc_processor"
    output_content_type_fn: Callable[[dict[str, Any]], str | None] | None = None
    """import 槽位宣告:依方法參數推導 content_type(如 local_file
    依 extensions 推導 text / pdf / mixed)。"""
    input_content_types: frozenset[str] = frozenset()  # parsing 槽位宣告
    produces_pages: bool = False  # parsing 槽位宣告
    requires_pages: bool = False  # chunking 槽位宣告
    capabilities: frozenset[str] = frozenset()  # indexing 槽位宣告
    required_capabilities: frozenset[str] = frozenset()  # retrieval 槽位宣告


@dataclass
class BuildContext:
    """建構期共享的執行環境(store、方法名稱、跨槽位依賴)。"""

    store: Any = None
    indexing_method: str = ""
    embedding_config: "MethodConfig | None" = None
    generation_config: "MethodConfig | None" = None


@dataclass
class SlotGraph:
    """factory 回傳的內部圖片段(多元件方法),由 builder 展開接線。

    retrieval(query_embedder + retriever + joiner)與 auto parsing
    (router + converters + joiner)共用同一形狀。
    """

    components: dict[str, Any]  # 相對名稱 → 元件
    connections: list[tuple[str, str]]  # 圖內接線("元件.socket" 相對名)
    inputs: dict[str, list[tuple[str, str]]] = dc_field(default_factory=dict)
    """外部輸入名(sources / meta / query)→ 圖內落點清單。"""
    output: str = ""  # 輸出 documents 的元件(相對名稱)


class BaseParams(BaseModel):
    """所有方法參數 schema 的基底:多打欄位直接報錯,不靜默忽略。"""

    model_config = ConfigDict(extra="forbid")


def validate_params(
    slot: str, method: str, params_cls: type[BaseParams], raw: dict[str, Any]
) -> Any:
    """以方法的 Params schema 驗證參數,錯誤時列出可接受的參數。"""
    try:
        return params_cls(**raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        accepted = ", ".join(sorted(params_cls.model_fields.keys())) or "(不接受任何參數)"
        raise ConfigError(
            f"模組 '{slot}' 方法 '{method}' 的參數不合法:{details}。"
            f"可接受的參數:{accepted}"
        ) from exc


def resolve(slot: str, table: dict[str, SlotFactory], method: str) -> SlotFactory:
    """從對映表取出方法的 factory;不存在時列出所有可用方法。"""
    if method not in table:
        raise UnknownMethodError(slot, method, list(table.keys()))
    return table[method]


def require_single(slot: str, cfg: "MethodConfig") -> str:
    """取出唯一的方法名稱;不支援鏈的槽位收到清單時報錯。"""
    methods = cfg.methods()
    if len(methods) > 1 and slot not in CHAINABLE_SLOTS:
        raise ConfigError(
            f"模組 '{slot}' 的輸入輸出型別不同,不支援方法鏈;"
            f"收到 method 清單 {methods},請指定單一方法"
        )
    return methods[0]

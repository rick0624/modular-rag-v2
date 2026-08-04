"""槽位契約:custom module 必須滿足的 socket 形狀(建構期驗證)。

與 :mod:`rag.compatibility` 的分工:compatibility 檢查「方法組合的語意」
(content_type / requires_pages / 索引能力,宣告式欄位);本模組檢查
「元件本身的 socket 形狀」(introspection)—— custom module 是使用者自寫
的 Haystack ``@component``,框架無法從宣告得知其輸入輸出,必須在建構期
直接檢視 ``__haystack_input__`` / ``__haystack_output__``。

契約原則(見 docs/interfaces.md):
- 槽位邊界只流 canonical 型別(queries / documents / route)。
- 契約要求的 sockets 必須存在且型別相容;**額外的 output sockets 允許**
  (自動進 trace),**額外的必填 input sockets 不允許**(執行期沒有
  任何上游會餵它,pipeline 必然缺輸入而失敗)。
- 格式轉換(如公司欄位 → Document.meta)是 custom module 內部的責任,
  不是契約的一部分。

本模組刻意不 import :mod:`rag.builder`(builder → custom → contracts,
反向依賴會成環)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haystack.dataclasses import ByteStream, ChatMessage, Document

from rag.errors import ConfigError

# import 輸出 / parsing 鏈首輸入的 sources 型別:與 Haystack converter 的
# sources 參數同款 union。契約用它當 receiver 側規格,custom 元件註記
# list[str] / list[Path] / list[ByteStream] 都相容;但**不可**改成
# list[Any] —— sender 端的 list[Any] 對任何具體型別都不相容(strict-Any)。
_SOURCES_TYPE = list[str | Path | ByteStream]
_META_TYPE = list[dict[str, Any]]


@dataclass(frozen=True)
class SocketSpec:
    """契約中一個 socket 的名稱與期望型別。"""

    name: str
    type: Any


@dataclass(frozen=True)
class SlotContract:
    """一個槽位對 custom 元件的 socket 要求。"""

    slot: str
    inputs: tuple[SocketSpec, ...]
    outputs: tuple[SocketSpec, ...]


SLOT_CONTRACTS: dict[str, SlotContract] = {
    "query_transformation": SlotContract(
        slot="query_transformation",
        inputs=(SocketSpec("queries", list[str]),),
        outputs=(SocketSpec("queries", list[str]),),
    ),
    "retrieval": SlotContract(
        slot="retrieval",
        inputs=(SocketSpec("query", str),),
        outputs=(SocketSpec("documents", list[Document]),),
    ),
    "reranking": SlotContract(
        slot="reranking",
        inputs=(
            SocketSpec("query", str),
            SocketSpec("documents", list[Document]),
        ),
        outputs=(SocketSpec("documents", list[Document]),),
    ),
    "routing": SlotContract(
        slot="routing",
        inputs=(SocketSpec("query", str),),
        outputs=(SocketSpec("route", dict[str, Any]),),
    ),
    # generation 的契約是 Haystack 的 ChatGenerator 形狀 —— prompt 的組裝
    # 仍由框架的 ChatPromptBuilder 負責(custom 元件收到的是組好的
    # messages),因此 custom generator 也能被 llm_rewrite / llm_decompose /
    # llm rerank / llm_fact_check 的 params.generator 沿用。
    "generation": SlotContract(
        slot="generation",
        inputs=(SocketSpec("messages", list[ChatMessage]),),
        outputs=(SocketSpec("replies", list[ChatMessage]),),
    ),
    # fusion:跨子查詢的合併步驟。掛了 custom 就一律執行(單一查詢的
    # N=1 也進元件,原樣通過與否由元件決定);建議額外輸出 applied: bool
    # (trace 用之區分「融合過」與「路過」,缺席時顯示為未回報)。
    "fusion": SlotContract(
        slot="fusion",
        inputs=(SocketSpec("results", list[list[Document]]),),
        outputs=(SocketSpec("documents", list[Document]),),
    ),
    # import:pipeline 的最上游,以 pipeline.run({}) 啟動 —— 沒有任何輸入
    # (空契約 + 「契約外必填輸入拒絕」= 任何必填輸入都會被擋)。
    # meta 的每個元素**必須帶 doc_id**(socket 驗不到;執行期由
    # ChunkMetaStamper 兜底報錯)。
    "import": SlotContract(
        slot="import",
        inputs=(),
        outputs=(
            SocketSpec("sources", _SOURCES_TYPE),
            SocketSpec("meta", _META_TYPE),
        ),
    ),
    # parsing 有兩種位置、兩份契約:鏈中的文件處理器(預設)與鏈首的
    # converter(params 設 kind: converter 時採用,見 "parsing_converter")。
    "parsing": SlotContract(
        slot="parsing",
        inputs=(SocketSpec("documents", list[Document]),),
        outputs=(SocketSpec("documents", list[Document]),),
    ),
    "parsing_converter": SlotContract(
        slot="parsing_converter",
        inputs=(
            SocketSpec("sources", _SOURCES_TYPE),
            SocketSpec("meta", _META_TYPE),
        ),
        outputs=(SocketSpec("documents", list[Document]),),
    ),
    "chunking": SlotContract(
        slot="chunking",
        inputs=(SocketSpec("documents", list[Document]),),
        outputs=(SocketSpec("documents", list[Document]),),
    ),
}


def _type_name(t: Any) -> str:
    """型別的可讀名稱(優先用 Haystack 自己的格式化)。"""
    try:
        from haystack.core.type_utils import _type_name as haystack_type_name

        return haystack_type_name(t)
    except Exception:
        return getattr(t, "__name__", None) or str(t)


def _types_compatible(sender: Any, receiver: Any) -> bool:
    """寬鬆型別相容(與 ``Pipeline.connect`` 同判準)。

    ``_types_are_compatible`` 是 Haystack 私有 API(2.31 回傳
    ``(bool, ConversionStrategy)`` tuple,舊版回傳 bool);包一層防禦,
    API 變動時退回字串比對而不是讓建構期驗證整個失效。
    """
    try:
        from haystack.core.type_utils import _types_are_compatible

        result = _types_are_compatible(sender, receiver)
        return result[0] if isinstance(result, tuple) else bool(result)
    except Exception:
        return _type_name(sender) == _type_name(receiver)


def _format_sockets(sockets: dict[str, Any]) -> str:
    """把 sockets dict 排成 ``name: type`` 清單(訊息用)。"""
    if not sockets:
        return "(沒有任何 socket)"
    return ", ".join(
        f"{name}: {_type_name(socket.type)}" for name, socket in sorted(sockets.items())
    )


def validate_component_contract(slot: str, instance: Any, *, where: str) -> None:
    """驗證 custom 元件滿足槽位契約;不滿足時以可行動的訊息報錯。

    Args:
        slot: 槽位名稱(必須存在於 :data:`SLOT_CONTRACTS`)。
        instance: 已實例化的元件。
        where: 錯誤訊息前綴,如
            ``"模組 'reranking' 方法 'custom'(類別 CompanyReranker)"``。

    Raises:
        ConfigError: 元件不是 Haystack component、缺少契約 sockets、
            型別不相容,或有契約外的必填輸入。
    """
    contract = SLOT_CONTRACTS.get(slot)
    if contract is None:
        raise ConfigError(
            f"{where}:槽位 '{slot}' 不支援 custom module。"
            f"支援的槽位:{', '.join(sorted(SLOT_CONTRACTS))}"
        )

    input_container = getattr(instance, "__haystack_input__", None)
    output_container = getattr(instance, "__haystack_output__", None)
    if input_container is None or output_container is None:
        raise ConfigError(
            f"{where}:類別 {type(instance).__name__} 不是 Haystack 元件。"
            "請在類別上加 @component 裝飾器,並以 "
            "@component.output_types(...) 宣告 run() 的輸出"
        )
    input_sockets: dict[str, Any] = dict(input_container._sockets_dict)
    output_sockets: dict[str, Any] = dict(output_container._sockets_dict)

    # 契約輸入:必須存在且型別相容(pipeline 送出契約型別 → 元件接收)。
    for spec in contract.inputs:
        socket = input_sockets.get(spec.name)
        if socket is None:
            raise ConfigError(
                f"{where}:缺少輸入 socket '{spec.name}: {_type_name(spec.type)}'。"
                f"實際的輸入 sockets:{_format_sockets(input_sockets)}。"
                f"請在 run() 加入參數 '{spec.name}'(型別 {_type_name(spec.type)})"
            )
        if not _types_compatible(spec.type, socket.type):
            raise ConfigError(
                f"{where}:輸入 socket '{spec.name}' 的型別是 "
                f"{_type_name(socket.type)},但槽位 '{slot}' 會送入 "
                f"{_type_name(spec.type)}。請把 run() 參數 '{spec.name}' 的"
                f"型別註記改為 {_type_name(spec.type)}"
            )

    # 契約外的必填輸入:圖上沒有任何上游會餵它,執行期必缺輸入。
    contract_input_names = {spec.name for spec in contract.inputs}
    extra_mandatory = sorted(
        name
        for name, socket in input_sockets.items()
        if name not in contract_input_names and socket.is_mandatory
    )
    if extra_mandatory:
        listed = ", ".join(repr(name) for name in extra_mandatory)
        provided = (
            f"槽位 '{slot}' 執行時只會提供 "
            f"{', '.join(repr(s.name) for s in contract.inputs)},"
            "其他輸入沒有上游可餵。"
            if contract.inputs
            else f"槽位 '{slot}' 不提供任何輸入"
            "(ingestion 以 pipeline.run({}) 啟動,元件必須能無輸入執行)。"
        )
        raise ConfigError(
            f"{where}:有契約以外的必填輸入 socket {listed}。"
            f"{provided}請在 run() 給這些參數預設值,"
            "或改為 __init__ 參數並經由 init_params 設定"
        )

    # 契約輸出:必須存在且型別相容(元件送出 → pipeline 下游接收契約型別)。
    for spec in contract.outputs:
        socket = output_sockets.get(spec.name)
        if socket is None:
            raise ConfigError(
                f"{where}:缺少輸出 socket '{spec.name}: {_type_name(spec.type)}'。"
                f"實際的輸出 sockets:{_format_sockets(output_sockets)}。"
                f"請在 @component.output_types(...) 宣告 "
                f"{spec.name}={_type_name(spec.type)},"
                f"並在 run() 回傳的 dict 中提供該 key"
            )
        if not _types_compatible(socket.type, spec.type):
            raise ConfigError(
                f"{where}:輸出 socket '{spec.name}' 的型別是 "
                f"{_type_name(socket.type)},但槽位 '{slot}' 的下游期望 "
                f"{_type_name(spec.type)}。請把 @component.output_types 中 "
                f"'{spec.name}' 的型別改為 {_type_name(spec.type)}"
            )
    # 額外的 output sockets 一律允許:query() 以 include_outputs_from 收集
    # 全部節點輸出,額外輸出自動進 trace,不需要接線。

"""custom module:載入使用者的 Haystack ``@component`` 並驗證 socket 契約。

使用者自寫一個 Haystack 元件,config 以

- ``class_path: "pkg.mod:Class"``(已安裝套件/可 import 的模組),或
- ``file: ./path/to/module.py`` + ``class: ClassName``(單一 .py 檔)

指定,``init_params`` 透傳給建構子。載入後由本檔的
:func:`validate_component_contract` 驗證 socket 契約,所有錯誤都在
建構期發生、訊息指明修法。

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

安全註記:custom module 就是執行任意 Python 程式碼 —— 與 config 檔
本身同一信任層級,只載入你信任的來源。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haystack.dataclasses import ByteStream, ChatMessage, Document
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.errors import ConfigError

logger = logging.getLogger(__name__)

# ``file:`` 載入的模組掛在這個虛擬套件底下(模組名 = 前綴 + 檔名 + 路徑雜湊)。
# :mod:`rag.logging_config` 會把這棵樹設成 DEBUG,使用者在 custom module 裡寫
# ``logging.getLogger(__name__)`` 才記得進 log 檔(見該模組的 setup_logging)。
CUSTOM_MODULE_PACKAGE = "_rag_custom"

# ---------------------------------------------------------------------------
# 槽位契約:custom 元件必須滿足的 socket 形狀
# ---------------------------------------------------------------------------

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
    # formatter:選填的終端支線 —— 把融合後結果組成對外格式(公司信封)。
    # ``payload: Any`` 是**終端槽位的特權**:圖上沒有下游要接它,型別開放
    # 不會斷任何接線,唯一消費者是 query()(原樣放進回傳值的 output 鍵)。
    # 中間槽位不可模仿 —— 邊界一開放,下游的契約檢查就全部失效。
    # payload 走 HTTP(/query 回應)時須為 JSON 可序列化。
    "formatter": SlotContract(
        slot="formatter",
        inputs=(
            SocketSpec("documents", list[Document]),
            SocketSpec("query", str),
        ),
        outputs=(SocketSpec("payload", Any),),
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
    except Exception as exc:
        logger.warning(
            "fail-soft:Haystack 型別相容 API 呼叫失敗(%s: %s),"
            "契約檢查退回字串比對(可能放過型別不符的 custom 元件)",
            type(exc).__name__, exc,
        )
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
        if spec.type is Any:
            # 終端槽位(formatter)的開放型別:socket 必須存在,但元件宣告
            # 什麼具體型別都接受。顯式跳過而不依賴 Haystack 私有 API 對
            # Any 的版本行為(strict-Any:sender 端 Any 對具體型別不相容)。
            continue
        if not _types_compatible(socket.type, spec.type):
            raise ConfigError(
                f"{where}:輸出 socket '{spec.name}' 的型別是 "
                f"{_type_name(socket.type)},但槽位 '{slot}' 的下游期望 "
                f"{_type_name(spec.type)}。請把 @component.output_types 中 "
                f"'{spec.name}' 的型別改為 {_type_name(spec.type)}"
            )
    # 額外的 output sockets 一律允許:query() 以 include_outputs_from 收集
    # 全部節點輸出,額外輸出自動進 trace,不需要接線。


class CustomModuleParams(BaseModel):
    """``method: custom`` 的參數 schema(所有支援 custom 的槽位共用)。

    不繼承 builder 的 BaseParams(builder → custom 的 import 方向不可反轉),
    但同樣 ``extra="forbid"``,且可直接餵給 builder 的 ``_validate_params``。
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_path: str | None = Field(
        default=None,
        description="元件類別的 import 路徑,格式 'pkg.mod:ClassName';"
        "與 file 擇一",
    )
    file: str | None = Field(
        default=None,
        description=".py 檔路徑(相對於執行目錄);需搭配 class 指定類別名稱",
    )
    class_: str | None = Field(
        default=None,
        alias="class",
        description="file 檔案中的類別名稱(僅搭配 file 使用)",
    )
    init_params: dict[str, Any] = Field(
        default_factory=dict, description="透傳給元件建構子的參數"
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "CustomModuleParams":
        if (self.class_path is None) == (self.file is None):
            raise ValueError(
                "class_path 與 file 必須恰好指定一個:"
                "已安裝的套件用 class_path('pkg.mod:ClassName'),"
                "單一 .py 檔用 file + class"
            )
        if self.file is not None and self.class_ is None:
            raise ValueError("使用 file 時必須以 class 指定檔案中的類別名稱")
        if self.class_path is not None and self.class_ is not None:
            raise ValueError(
                "class 只能搭配 file 使用;class_path 已含類別名稱"
                "('pkg.mod:ClassName')"
            )
        return self


def _public_classes(module: Any) -> list[str]:
    """列出模組中定義的公開類別名稱(錯誤訊息用)。"""
    return sorted(
        name
        for name, obj in vars(module).items()
        if inspect.isclass(obj)
        and not name.startswith("_")
        and getattr(obj, "__module__", None) == module.__name__
    )


def _class_from_module(where: str, module: Any, cls_name: str, origin: str) -> type:
    """從模組取出類別;不存在時列出實際可用的類別。"""
    cls = getattr(module, cls_name, None)
    if not inspect.isclass(cls):
        available = _public_classes(module)
        listed = (
            ", ".join(repr(name) for name in available)
            if available
            else "(沒有任何公開類別)"
        )
        raise ConfigError(
            f"{where}:{origin} 中沒有名為 '{cls_name}' 的類別。"
            f"實際定義的類別:{listed}"
        )
    return cls


def _load_from_class_path(where: str, class_path: str) -> type:
    module_path, sep, cls_name = class_path.partition(":")
    if not sep or not module_path or not cls_name:
        raise ConfigError(
            f"{where}:class_path '{class_path}' 格式不正確。"
            "期望 'pkg.mod:ClassName'(模組路徑與類別名稱以冒號分隔)"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigError(
            f"{where}:無法 import 模組 '{module_path}'"
            f"({type(exc).__name__}: {exc})。"
            "請確認套件已安裝且模組在 PYTHONPATH 上;"
            "單一 .py 檔請改用 file + class 指定"
        ) from exc
    return _class_from_module(where, module, cls_name, f"模組 '{module_path}'")


def _load_from_file(where: str, file: str, cls_name: str) -> type:
    path = Path(file)
    if not path.is_file():
        raise ConfigError(
            f"{where}:找不到檔案 '{path.resolve()}'"
            f"(目前執行目錄:{Path.cwd()})。"
            "file 路徑相對於執行目錄解析,不是 config 檔的位置"
        )
    resolved = str(path.resolve())
    # 模組名帶路徑雜湊:不同路徑的同名檔不互撞;每次建構重新 exec,
    # /reload 換了檔案內容即生效。
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:8]
    module_name = f"{CUSTOM_MODULE_PACKAGE}.{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ConfigError(f"{where}:無法從 '{resolved}' 建立模組載入器")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # exec 前登記:支援檔內 dataclass 等自參照
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(
            f"{where}:載入 '{resolved}' 時發生錯誤"
            f"({type(exc).__name__}: {exc})"
        ) from exc
    return _class_from_module(where, module, cls_name, f"檔案 '{path}'")


def instantiate_custom(
    slot: str,
    params: CustomModuleParams,
    *,
    method: str = "custom",
    contract: str | None = None,
) -> Any:
    """載入、實例化並驗證 custom 元件(建構期一站完成)。

    Args:
        slot: 槽位名稱(進錯誤訊息前綴)。
        params: 已驗證的 custom module 參數。
        method: 方法名稱(進錯誤訊息前綴)。
        contract: 用哪份 :data:`SLOT_CONTRACTS` 契約驗證;
            None 時用 ``slot`` 本身。同一槽位有多種形狀時用
            (parsing 依 kind 選 "parsing" / "parsing_converter",
            錯誤訊息前綴仍是槽位名稱)。

    Raises:
        ConfigError: 載入失敗、建構子參數不符,或 socket 契約不滿足。
    """
    source = params.class_path or f"{params.file}::{params.class_}"
    where = f"模組 '{slot}' 方法 '{method}'({source})"
    if params.class_path is not None:
        cls = _load_from_class_path(where, params.class_path)
    else:
        cls = _load_from_file(where, params.file, params.class_)

    try:
        instance = cls(**params.init_params)
    except TypeError as exc:
        try:
            signature = str(inspect.signature(cls.__init__))
        except (TypeError, ValueError):
            signature = "(無法取得)"
        raise ConfigError(
            f"{where}:以 init_params 建構 {cls.__name__} 失敗"
            f"({exc})。建構子簽名:{cls.__name__}.__init__{signature}。"
            "請對齊 init_params 的鍵與建構子參數"
        ) from exc

    validate_component_contract(
        contract or slot,
        instance,
        where=f"模組 '{slot}' 方法 '{method}'(類別 {cls.__name__})",
    )
    return instance

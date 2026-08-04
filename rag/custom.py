"""custom module 載入:把使用者的 Haystack ``@component`` 掛進槽位。

custom module 是「擴充階梯」的中間層:內建方法(config 選項)不夠用、
但又不需要整條 escape hatch 時,使用者自寫一個 Haystack 元件,config 以

- ``class_path: "pkg.mod:Class"``(已安裝套件/可 import 的模組),或
- ``file: ./path/to/module.py`` + ``class: ClassName``(單一 .py 檔)

指定,``init_params`` 透傳給建構子。載入後由
:func:`rag.contracts.validate_component_contract` 驗證 socket 契約,
所有錯誤都在建構期發生、訊息指明修法。

安全註記:custom module 就是執行任意 Python 程式碼 —— 與 config 檔
本身同一信任層級,只載入你信任的來源。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.contracts import validate_component_contract
from rag.errors import ConfigError


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
    module_name = f"_rag_custom.{path.stem}_{digest}"
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
    slot: str, params: CustomModuleParams, *, method: str = "custom"
) -> Any:
    """載入、實例化並驗證 custom 元件(建構期一站完成)。

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
        slot, instance, where=f"模組 '{slot}' 方法 '{method}'(類別 {cls.__name__})"
    )
    return instance

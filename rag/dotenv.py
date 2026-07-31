""".env 檔的解析與載入(不依賴 python-dotenv)。

支援的格式::

    # 註解與空行會被忽略
    KEY=value
    export KEY=value          # 允許 export 前綴
    KEY="含 空白 的值"         # 單/雙引號包裹的值,引號會被去除
    KEY=value  # 未加引號的值支援行內註解(以「空白 + #」分隔)

已存在的環境變數預設不會被覆蓋(真正的環境變數優先於 .env)。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from rag.errors import ConfigError

_LINE_PATTERN = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def parse_dotenv(text: str, source: str = ".env") -> dict[str, str]:
    """把 .env 內容解析成 dict(不寫入環境變數)。

    Args:
        text: .env 檔的文字內容。
        source: 來源描述,用於錯誤訊息。

    Raises:
        ConfigError: 任一行不是 ``KEY=value`` 格式。
    """
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE_PATTERN.match(line)
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

    Args:
        path: .env 檔路徑(預設為目前工作目錄下的 ``.env``)。
        override: 是否覆蓋已存在的環境變數;預設不覆蓋,
            讓真正的環境變數(export / 系統設定)優先。

    Returns:
        實際寫入環境的變數 dict(被跳過的不含在內)。

    Raises:
        ConfigError: 檔案存在但內容格式不正確。
    """
    env_path = Path(path)
    if not env_path.is_file():
        return {}
    values = parse_dotenv(env_path.read_text(encoding="utf-8"), source=str(env_path))
    applied: dict[str, str] = {}
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied

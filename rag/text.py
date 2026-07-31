"""共用的輕量文字處理工具(不依賴外部套件)。

中文(CJK)不依賴空白斷詞,改用字元 bigram;供 mock embedding 等
需要確定性 token 的元件共用。
"""

from __future__ import annotations

import re

# 拉丁字母 / 數字組成的詞,或單一 CJK 字元。
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[一-鿿㐀-䶿豈-﫿]")


def tokenize(text: str) -> list[str]:
    """把文字切成小寫 token 清單。

    規則:

    - 拉丁字母 / 數字連續段落視為一個詞(轉小寫)。
    - CJK 字元以「相鄰兩字 bigram」為 token(連續 CJK 段落內滑動);
      孤立的單一 CJK 字元則以單字為 token。

    Args:
        text: 任意文字。

    Returns:
        token 清單(可能為空)。
    """
    raw = _TOKEN_PATTERN.findall(text)
    tokens: list[str] = []
    cjk_run: list[str] = []

    def _flush_cjk() -> None:
        if not cjk_run:
            return
        if len(cjk_run) == 1:
            tokens.append(cjk_run[0])
        else:
            tokens.extend(cjk_run[i] + cjk_run[i + 1] for i in range(len(cjk_run) - 1))
        cjk_run.clear()

    for piece in raw:
        if len(piece) == 1 and not piece.isascii():
            cjk_run.append(piece)
        else:
            _flush_cjk()
            tokens.append(piece.lower())
    _flush_cjk()
    return tokens

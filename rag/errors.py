"""框架層級的例外定義。

所有例外都繼承自 :class:`RAGFrameworkError`。訊息一律遵循框架的
錯誤訊息守則:指出收到什麼、期望什麼、該調整哪個 config 欄位,
並在可能時列出可用的替代選項。
"""

from __future__ import annotations

from collections.abc import Sequence


class RAGFrameworkError(Exception):
    """框架內所有自訂例外的共同基底。"""


class ConfigError(RAGFrameworkError):
    """config 載入或驗證失敗(欄位缺漏、型別錯誤、參數不合法等)。"""


class UnknownMethodError(ConfigError):
    """config 指定的 method 不在該槽位的 factory 表中。

    訊息會列出該槽位所有可用的方法。
    """

    def __init__(self, slot: str, method: str, available: Sequence[str]) -> None:
        self.slot = slot
        self.method = method
        self.available = sorted(available)
        listed = ", ".join(repr(name) for name in self.available) or "(目前沒有任何可用的方法)"
        super().__init__(
            f"模組 '{slot}' 沒有名為 '{method}' 的方法。可用的方法:{listed}"
        )


class IncompatiblePipelineError(ConfigError):
    """config 指定的方法組合不相容(例如 PDF importer 搭配純文字 parser)。

    在 pipeline 建構期即檢查,訊息指出不相容的兩個槽位與可相容的方法。
    """


class ComponentError(RAGFrameworkError):
    """元件在執行期間發生的錯誤(檔案不存在、資料格式不符等)。"""


class APICallError(ComponentError):
    """呼叫外部 API 失敗(timeout、連線錯誤、非 2xx 狀態碼)。"""


class APIResponseFormatError(APICallError):
    """外部 API 回傳格式不符預期(非 JSON、缺少欄位、型別不對)。"""


class MissingDependencyError(RAGFrameworkError):
    """選用相依套件未安裝(例如 sentence-transformers、elasticsearch)。"""

    def __init__(self, package: str, purpose: str) -> None:
        super().__init__(
            f"缺少相依套件 '{package}'({purpose})。請先執行:pip install {package}"
        )

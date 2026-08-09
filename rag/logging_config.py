"""日誌設定:終端機看重點,log 檔留完整過程。

兩路分開的理由 —— 終端機要能讀(只印警告與明確要求的內容),但事後
追問題時需要的是完整紀錄:每次 LLM 的 prompt 與回覆、每個 HTTP 請求、
每一步的中間結果。同一份輸出滿足不了這兩種需求,所以用兩個 handler
各自設層級。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_CONSOLE_FORMAT = "%(levelname)s %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# 每次執行必定出現、且與本專案無關的警告(HF_TOKEN 未設、Windows 不支援
# symlink…)。只擋終端機,log 檔照收 —— 真要查的時候還是找得到。
_CONSOLE_MUTED = ("huggingface_hub", "py.warnings")

# 有 log 檔時額外收進檔案的第三方紀錄:HTTP 請求、ES 查詢、Haystack 的
# 元件執行順序。**只點名這幾個**,不是把 root 降到 INFO ——
# sentence-transformers 會看 logger 的 effective level 決定要不要顯示
# 進度條,root 一降級終端機就跑出一堆 "Batches: 100%|…"。
_FILE_VERBOSE = ("httpx", "openai", "elastic_transport", "haystack")

# 與本專案同等對待(一律 DEBUG)的 logger 樹:
# - "rag":框架自己的元件,以及刻意掛在底下的 CLI logger(rag.demo…)。
# - "_rag_custom":``file:`` 載入的 custom module —— 模組名由
#   :data:`rag.custom.CUSTOM_MODULE_PACKAGE` 決定,不在 "rag" 底下,
#   沒有這一條的話使用者寫的 ``logging.getLogger(__name__)`` 會繼承 root
#   的層級(預設 WARNING),INFO / DEBUG 在發出當下就被丟掉,連 log 檔的
#   DEBUG handler 都輪不到 —— 靜默失效,最難查。
#   ``class_path:`` 載入的模組名稱是任意套件路徑,框架蓋不到,慣例是把
#   logger 命名在 "rag.custom.*" 底下(見 README「自訂方法」)。
_PROJECT_LOGGERS = ("rag", "_rag_custom")


class _MuteNoisyDeps(logging.Filter):
    """終端機端過濾:指定套件低於 ERROR 的訊息不印。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        return not record.name.startswith(_CONSOLE_MUTED)


class _WarningTally(logging.Handler):
    """收集本專案 logger 樹的 WARNING+ 訊息,供執行結束時彙報。

    fail-soft 機制(API 掛掉保留原順序、LLM 故障退回原查詢…)讓流程
    「看似成功」;個別警告會即時印出,但長輸出裡容易被滑掉。這個
    handler 把它們累積起來,結束時一次總結:「這次執行有幾件事降級了」。
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}: {record.getMessage()}")


_tally: _WarningTally | None = None


def warning_tally() -> list[str]:
    """回傳自上次 :func:`setup_logging` 以來,本專案發出的 WARNING+ 訊息。

    只收 ``rag.*`` 與 ``_rag_custom.*``(第三方套件的警告不算),依發生
    順序;供 CLI 在執行結束時總結「有哪些步驟降級了」。
    """
    return list(_tally.messages) if _tally is not None else []


def default_log_path(directory: str | Path = "logs") -> Path:
    """產生帶時間戳的預設路徑,避免多次執行互相覆蓋。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(directory) / f"run-{stamp}.log"


def quiet_dependency_handlers() -> None:
    """移除套件自己裝的 console handler,讓訊息回到 root 走正常流程。

    ``huggingface_hub`` 會在 import 時裝一個沒有 formatter 的
    StreamHandler,直接寫 stderr —— 它繞過 root 上的過濾器,所以
    「沒設 HF_TOKEN」那行每次執行都會冒出來。清掉之後訊息改由 root
    處理:終端機被 :class:`_MuteNoisyDeps` 擋下,log 檔照樣收得到。

    這些套件多半是**延遲載入**的(建 pipeline 時才 import),
    :func:`setup_logging` 當下可能還沒有 handler 可清,因此建好
    pipeline 之後可以再呼叫一次(重複呼叫安全)。
    """
    for name in _CONSOLE_MUTED:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.propagate = True


def setup_logging(
    log_file: str | Path | None = None,
    console_level: str = "WARNING",
) -> Path | None:
    """設定終端機與(選填)檔案兩路日誌。

    ``rag.*`` 與 ``_rag_custom.*``(``file:`` 載入的 custom module)一律
    開到 DEBUG(LLM 的 prompt / 回覆就在這一層),由各 handler 自己的層級
    決定印不印;第三方套件在有 log 檔時開到 INFO —— httpx 的請求紀錄有用,
    但它們的 DEBUG 會把檔案灌爆。

    Args:
        log_file: log 檔路徑;``None`` 表示不寫檔。父目錄會自動建立。
        console_level: 終端機層級(DEBUG / INFO / WARNING / ERROR)。

    Returns:
        實際寫入的 log 檔路徑(``log_file`` 為 None 時回傳 None)。
    """
    global _tally
    root = logging.getLogger()
    for handler in list(root.handlers):  # 重複呼叫不要疊 handler(訊息會重複)
        root.removeHandler(handler)
        handler.close()
    root.setLevel(getattr(logging, console_level))
    _tally = _WarningTally()  # 每次 setup 重置(舊 handler 一併換掉)
    for name in _PROJECT_LOGGERS:
        project_logger = logging.getLogger(name)
        project_logger.setLevel(logging.DEBUG)
        for handler in list(project_logger.handlers):
            if isinstance(handler, _WarningTally):
                project_logger.removeHandler(handler)
        project_logger.addHandler(_tally)
    for name in _FILE_VERBOSE:
        logging.getLogger(name).setLevel(logging.INFO if log_file else logging.NOTSET)

    logging.captureWarnings(True)  # warnings.warn 也收進來(py.warnings)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, console_level))
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    console.addFilter(_MuteNoisyDeps())
    root.addHandler(console)

    quiet_dependency_handlers()
    if log_file is None:
        return None
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")  # 中文不可用系統編碼
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    root.addHandler(file_handler)
    return path

"""log 檔設定測試:終端機看重點、檔案留完整過程。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from rag.custom import CUSTOM_MODULE_PACKAGE, CustomModuleParams, instantiate_custom
from rag.logging_config import (
    _PROJECT_LOGGERS,
    _WarningTally,
    default_log_path,
    setup_logging,
    warning_tally,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _restore_logging():
    """測試會動到 root handler,跑完還原,避免污染其他測試。"""
    root = logging.getLogger()
    saved_levels = {name: logging.getLogger(name).level for name in _PROJECT_LOGGERS}
    saved = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.handlers, root.level = saved[0], saved[1]
    for name, level in saved_levels.items():
        project = logging.getLogger(name)
        project.setLevel(level)
        for handler in list(project.handlers):  # 摘掉殘留的警告彙整 handler
            if isinstance(handler, _WarningTally):
                project.removeHandler(handler)


def test_file_gets_debug_while_console_stays_quiet(tmp_path, capsys):
    path = setup_logging(tmp_path / "run.log", console_level="WARNING")
    logger = logging.getLogger("rag.demo")

    logger.debug("prompt 內容")
    logger.info("步驟紀錄")
    logger.warning("出事了")

    content = path.read_text(encoding="utf-8")
    assert "prompt 內容" in content  # DEBUG 只進檔案
    assert "步驟紀錄" in content
    assert "出事了" in content
    console = capsys.readouterr().err
    assert "prompt 內容" not in console
    assert "步驟紀錄" not in console
    assert "出事了" in console  # 終端機只留 WARNING 以上


def test_log_file_parent_directory_is_created(tmp_path):
    path = setup_logging(tmp_path / "nested" / "dir" / "run.log")
    logging.getLogger("rag.demo").info("hi")
    assert path.is_file()


def test_no_log_file_returns_none(tmp_path):
    assert setup_logging(None) is None


def test_repeated_setup_does_not_duplicate_messages(tmp_path):
    """重複呼叫不可疊 handler —— 否則每行訊息會被寫兩次。"""
    setup_logging(tmp_path / "first.log")
    path = setup_logging(tmp_path / "second.log")
    logging.getLogger("rag.demo").info("只該出現一次")

    assert path.read_text(encoding="utf-8").count("只該出現一次") == 1
    assert (tmp_path / "first.log").read_text(encoding="utf-8") == ""


def test_noisy_dependencies_are_muted_on_console_only(tmp_path, capsys):
    """HF_TOKEN 之類的警告每次執行都出現,終端機擋掉但 log 檔要留。"""
    path = setup_logging(tmp_path / "run.log")

    logging.getLogger("huggingface_hub.utils._http").warning("沒設 HF_TOKEN")
    logging.getLogger("rag.components.fact_check").warning("查核失敗")

    console = capsys.readouterr().err
    assert "沒設 HF_TOKEN" not in console
    assert "查核失敗" in console  # 本專案的警告照印
    content = path.read_text(encoding="utf-8")
    assert "沒設 HF_TOKEN" in content and "查核失敗" in content


def test_muted_dependency_errors_still_reach_console(tmp_path, capsys):
    """只擋警告等級;ERROR 以上照印,否則真的壞掉會沒聲音。"""
    setup_logging(tmp_path / "run.log")
    logging.getLogger("huggingface_hub").error("下載失敗")
    assert "下載失敗" in capsys.readouterr().err


class TestCustomModuleLogging:
    """custom module 的紀錄要進 log 檔 —— 靜默失效是最難查的問題。"""

    def test_project_logger_list_matches_custom_module_package(self):
        """對映 rag.custom 的模組命名;改了那邊沒改這邊 = 紀錄再次消失。"""
        assert CUSTOM_MODULE_PACKAGE in _PROJECT_LOGGERS

    def test_file_loaded_module_logger_reaches_log_file(self, tmp_path, capsys):
        """``file:`` 載入的模組寫 getLogger(__name__),名字在 _rag_custom.* 底下。

        沒有把這棵樹設成 DEBUG 的話,INFO 會繼承 root 的 WARNING 在發出當下
        就被丟掉,連 log 檔的 DEBUG handler 都輪不到。
        """
        path = setup_logging(tmp_path / "run.log", console_level="WARNING")
        logger = logging.getLogger(f"{CUSTOM_MODULE_PACKAGE}.my_transform_ab12cd34")

        logger.debug("每條查詢的改寫結果")
        logger.info("custom module 收到 2 條查詢")

        content = path.read_text(encoding="utf-8")
        assert "custom module 收到 2 條查詢" in content
        assert "每條查詢的改寫結果" in content
        # 終端機照舊只留 WARNING 以上(log 檔完整 ≠ 終端機吵)
        assert "custom module 收到 2 條查詢" not in capsys.readouterr().err

    def test_rag_custom_naming_convention_also_works(self, tmp_path):
        """``class_path:`` 載入時模組名任意,慣例是自己命名在 rag.custom.* 底下。"""
        path = setup_logging(tmp_path / "run.log")
        logging.getLogger("rag.custom.query_expander").info("擴充後的查詢")
        assert "擴充後的查詢" in path.read_text(encoding="utf-8")

    def test_real_example_module_logs(self, tmp_path, monkeypatch):
        """實地驗一次 examples/ 的骨架:載入後呼叫 run(),紀錄要出現在檔案裡。"""
        monkeypatch.chdir(REPO_ROOT)
        path = setup_logging(tmp_path / "run.log")
        component = instantiate_custom(
            "query_transformation",
            CustomModuleParams(
                file="examples/custom_modules/company_query_transform.py",
                **{"class": "CompanyQueryExpander"},
            ),
        )
        component.run(queries=["請假規則?"])
        assert "CompanyQueryExpander" in path.read_text(encoding="utf-8")


class TestWarningTally:
    """警告彙整:fail-soft 讓流程「看似成功」,結束時要能總結有幾件事降級。"""

    def test_collects_project_warnings_in_order(self):
        setup_logging(None)
        logging.getLogger("rag.components.api_clients").warning(
            "fail-soft:rerank API 失敗"
        )
        logging.getLogger("_rag_custom.my_mod").warning("自訂元件警告")
        logging.getLogger("rag.demo").info("INFO 不列入")
        logging.getLogger("httpx").warning("第三方不列入")

        tally = warning_tally()
        assert len(tally) == 2
        assert "rerank API 失敗" in tally[0]
        assert "自訂元件警告" in tally[1]

    def test_reset_on_each_setup(self):
        setup_logging(None)
        logging.getLogger("rag.demo").warning("第一輪的警告")
        setup_logging(None)
        assert warning_tally() == []

    def test_repeated_setup_does_not_double_count(self):
        setup_logging(None)
        setup_logging(None)
        logging.getLogger("rag.demo").warning("只該算一次")
        assert len(warning_tally()) == 1


def test_default_path_is_timestamped_under_logs():
    path = default_log_path()
    assert path.parent.name == "logs"
    assert path.name.startswith("run-") and path.suffix == ".log"

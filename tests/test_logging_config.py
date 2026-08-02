"""log 檔設定測試:終端機看重點、檔案留完整過程。"""

from __future__ import annotations

import logging

import pytest

from rag.logging_config import default_log_path, setup_logging


@pytest.fixture(autouse=True)
def _restore_logging():
    """測試會動到 root handler,跑完還原,避免污染其他測試。"""
    root = logging.getLogger()
    saved = list(root.handlers), root.level, logging.getLogger("rag").level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.handlers, root.level = saved[0], saved[1]
    logging.getLogger("rag").setLevel(saved[2])


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


def test_default_path_is_timestamped_under_logs():
    path = default_log_path()
    assert path.parent.name == "logs"
    assert path.name.startswith("run-") and path.suffix == ".log"

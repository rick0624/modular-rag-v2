"""檔案層增量:內容沒變的來源檔案連 parse 都跳過。

切片層的 :class:`~rag.components.change_filter.IncrementalChangeFilter`
只省 embedding —— parsing 仍是全量。文字檔無所謂,但 OCR 每頁要數秒,
51 頁的 PDF 每次 ingest 都重跑一次是不可用的。本元件放在 importer 之後:
以檔案 bytes 的 sha256 對照上次 ingest 留下的 manifest,沒變的檔案直接
從 ``sources`` / ``meta`` 清單移除,下游(含 OCR)完全不會看到它們。

manifest 帶 ``key``(parse 設定的雜湊):parsing / chunking / OCR 設定
變了,同一份檔案會產出不同切片,舊 manifest 整份作廢、全量重 parse
—— 否則「檔案沒變就跳過」會讓索引內容停留在舊設定。

輸出的 ``manifest`` 是**這次全部**可雜湊檔案的雜湊(含被跳過者),由
``RagPipelines.run_ingestion`` 在成功後持久化;從資料夾刪除的檔案自然
不在其中,之後重新加回會被視為新檔案。

已知限制:檔案層比對只對**本地路徑** sources 有效。custom importer 回傳
ByteStream / API 參照時無從讀檔計雜湊,這些來源一律視為已變更(每次都
重 parse)且不記入 manifest —— 切片層增量(IncrementalChangeFilter)仍
會跳過內容未變的 embedding,所以退化是「省不到 parse」而不是全失效。
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable

from haystack import component
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)


@component
class SourceChangeFilter:
    """按檔案內容雜湊過濾 sources,只放行新增或變更的檔案。"""

    def __init__(
        self,
        read_previous: Callable[[], dict[str, Any] | None],
        config_key: str,
    ) -> None:
        """
        Args:
            read_previous: 讀取上次 manifest 的 callable(執行期呼叫,
                同一 pipeline 實例重複 run 也拿得到最新狀態)。
            config_key: 目前 parse 設定的雜湊;與 manifest 的 ``key``
                不符時視同沒有 manifest(全量重 parse)。
        """
        self._read_previous = read_previous
        self.config_key = config_key

    # 輸入收 list[Any](custom importer 可能給 ByteStream);輸出宣告
    # **必須**是具體 union 而非 list[Any] —— sender 端的 list[Any] 對
    # 任何具體型別都不相容,會斷掉與下游 parser 的接線。
    @component.output_types(
        sources=list[str | Path | ByteStream],
        meta=list[dict[str, Any]],
        manifest=dict[str, Any],
        skipped_files=list[str],
    )
    def run(self, sources: list[Any], meta: list[dict[str, Any]]) -> dict[str, Any]:
        previous = self._read_previous() or {}
        previous_files: dict[str, str] = (
            previous.get("files", {}) if previous.get("key") == self.config_key else {}
        )

        current_files: dict[str, str] = {}
        kept_sources: list[Any] = []
        kept_meta: list[dict[str, Any]] = []
        skipped: list[str] = []
        unhashable = 0
        for source, entry in zip(sources, meta):
            doc_id = entry.get("doc_id") or str(source)
            digest: str | None
            try:
                digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            except OSError as exc:  # 列出後被移走等競態:當作變更,交給下游處理
                logger.warning("無法讀取 %s 計算雜湊(%s),視為已變更", source, exc)
                digest = None
            except (TypeError, ValueError):  # 非路徑(ByteStream / API 參照)
                unhashable += 1
                digest = None
            # 雜湊不了的來源**不記入 manifest**:記一個固定 sentinel 會讓
            # 下一輪誤判「沒變」而跳過 —— 寧可每次重 parse。
            if digest is None:
                kept_sources.append(source)
                kept_meta.append(entry)
                continue
            current_files[doc_id] = digest
            if previous_files.get(doc_id) == digest:
                skipped.append(doc_id)
            else:
                kept_sources.append(source)
                kept_meta.append(entry)

        if unhashable:
            logger.warning(
                "%d 個來源不是本地檔案路徑(custom importer?),無法做檔案層"
                "比對,一律重新 parse(切片層增量仍會跳過內容未變的 embedding)",
                unhashable,
            )
        if skipped:
            logger.info(
                "檔案層增量:跳過 %d/%d 個未變更的檔案(不重新 parse):%s%s",
                len(skipped), len(sources), ", ".join(skipped[:5]),
                "…" if len(skipped) > 5 else "",
            )
        return {
            "sources": kept_sources,
            "meta": kept_meta,
            "manifest": {"key": self.config_key, "files": current_files},
            "skipped_files": skipped,
        }

"""Ingestion pipeline 的框架步驟:匯入、檔案層增量、身分蓋章、切片層增量。

依 pipeline 順序:

- :class:`FileLister`(import 槽位):列出本地檔案並產生穩定的 doc_id。
- :class:`SourceChangeFilter`(``incremental: true``):檔案層增量 ——
  內容沒變的來源檔案連 parse(含 OCR)都跳過。
- :class:`ChunkMetaStamper`(內建步驟):蓋 doc_id / seq / page / chunk_id,
  upsert 語意的基石。
- :class:`IncrementalChangeFilter`(``incremental: true``):切片層增量 ——
  內容沒變的切片跳過 embedding 與寫入。
- :class:`FieldSourceEmbedder`(``embedding.source_field`` 非 content 或
  設定 ``extra_vectors`` 時):讓文件端 embedder 改用指定 meta 欄位當
  輸入,並可對額外欄位各出一組向量寫進 meta。
- :class:`MetaFieldMapper`(``indexing.params.fields`` 設定時):寫入前
  依白名單挑選/改名自訂 meta 欄位。

doc_id 契約:FileLister 在 pipeline 最上游決定 doc_id(檔案相對
``input_dir`` 的 POSIX 路徑),ChunkMetaStamper 消費它產生
``chunk_id = "{doc_id}::chunk_{seq}"`` —— 同一輸入永遠得到同一 id,
document store 的 OVERWRITE 寫入因此具備 upsert 語意,ES 的 ``_id``
也隨之穩定。
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable

from haystack import Document, component
from haystack.dataclasses import ByteStream

from rag.errors import ComponentError

logger = logging.getLogger(__name__)


@component
class FileLister:
    """列出資料夾中的檔案,輸出 converter 需要的 ``sources`` 與 ``meta``。

    Haystack 的 converter 會把 ``meta`` 清單逐一合併進產出的 Document,
    因此 doc_id 在這裡(pipeline 的最上游)就決定,不受 converter 的
    ``store_full_path`` 行為影響。同一來源每次執行都得到相同的 id
    (service/upsert 語意與評估資料集都依賴這一點)。
    """

    def __init__(
        self,
        input_dir: str,
        extensions: list[str],
        method_name: str = "local_file",
        recursive: bool = True,
    ) -> None:
        self.input_dir = input_dir
        self.extensions = {ext.lower() for ext in extensions}
        self.method_name = method_name
        self.recursive = recursive

    @component.output_types(sources=list[str], meta=list[dict[str, Any]])
    def run(self) -> dict[str, Any]:
        """掃描資料夾並回傳排序穩定的檔案清單。

        Raises:
            ComponentError: ``input_dir`` 不存在或不是資料夾。
        """
        base = Path(self.input_dir)
        if not base.is_dir():
            raise ComponentError(f"input_dir 不存在或不是資料夾:{base}")
        iterator = base.rglob("*") if self.recursive else base.glob("*")
        files = sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in self.extensions
        )
        meta = [
            {
                "doc_id": path.relative_to(base).as_posix(),
                "source": str(path),
                "importer": self.method_name,
            }
            for path in files
        ]
        return {"sources": [str(path) for path in files], "meta": meta}


@component
class SourceChangeFilter:
    """按檔案內容雜湊過濾 sources,只放行新增或變更的檔案(檔案層增量)。

    切片層的 :class:`IncrementalChangeFilter` 只省 embedding —— parsing
    仍是全量。文字檔無所謂,但 OCR 每頁要數秒,51 頁的 PDF 每次 ingest
    都重跑一次是不可用的。本元件放在 importer 之後:以檔案 bytes 的
    sha256 對照上次 ingest 留下的 manifest,沒變的檔案直接從 ``sources``
    / ``meta`` 清單移除,下游(含 OCR)完全不會看到它們。

    manifest 帶 ``key``(parse 設定的雜湊):parsing / chunking / OCR 設定
    變了,同一份檔案會產出不同切片,舊 manifest 整份作廢、全量重 parse
    —— 否則「檔案沒變就跳過」會讓索引內容停留在舊設定。

    輸出的 ``manifest`` 是**這次全部**可雜湊檔案的雜湊(含被跳過者),由
    ``RagPipelines.run_ingestion`` 在成功後持久化;從資料夾刪除的檔案自然
    不在其中,之後重新加回會被視為新檔案。

    已知限制:檔案層比對只對**本地路徑** sources 有效。custom importer
    回傳 ByteStream / API 參照時無從讀檔計雜湊,這些來源一律視為已變更
    (每次都重 parse)且不記入 manifest —— 切片層增量仍會跳過內容未變的
    embedding,所以退化是「省不到 parse」而不是全失效。
    """

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


@component
class ChunkMetaStamper:
    """為每個切片蓋上 doc_id / seq / page / chunk_id,並過濾空白切片。

    Haystack 的 ``Document.id`` 預設是內容雜湊;本框架改為顯式指定
    ``chunk_id = "{doc_id}::chunk_{seq}"``(v1 慣例)。放在 splitter
    之後、embedder 之前;此後不得再改動切片內容。
    """

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """依文件內出現順序編 seq,重建帶穩定 id 的 Document。

        規則(v1 契約):

        - ``seq`` 是**全文件**連續序號(不分頁重排)。
        - 空白切片不產生輸出(頁碼由 splitter 先數好,不受影響)。
        - meta 繼承上游全部欄位,再蓋上 doc_id / seq / page / chunk_id。

        Raises:
            ComponentError: 文件缺少 ``meta['doc_id']``
                (import 槽位的 FileLister 應在最上游提供)。
        """
        seq_counters: dict[str, int] = {}
        stamped: list[Document] = []
        for doc in documents:
            content = (doc.content or "").strip("\f \t\r\n")
            if not content:
                continue
            doc_id = doc.meta.get("doc_id")
            if not doc_id:
                raise ComponentError(
                    "切片缺少 meta['doc_id'],無法產生穩定的 chunk_id。"
                    "請確認 import 槽位使用會提供 doc_id 的方法(如 local_file),"
                    f"目前的 meta 欄位:{sorted(doc.meta.keys())}"
                )
            seq = seq_counters.get(doc_id, 0)
            seq_counters[doc_id] = seq + 1
            chunk_id = f"{doc_id}::chunk_{seq}"
            page = doc.meta.get("page_number")
            meta = {
                **doc.meta,
                "doc_id": doc_id,
                "seq": seq,
                "page": page if isinstance(page, int) else None,
                "chunk_id": chunk_id,
            }
            stamped.append(Document(id=chunk_id, content=content, meta=meta))
        return {"documents": stamped}


@component
class IncrementalChangeFilter:
    """按 chunk_id 比對 store 既有內容,只放行新增或變更的切片(切片層增量)。

    放在 stamper 之後、embedder 之前 —— 重點是**省下 embedding**:
    chunk_id 是確定性的,拿全部切片的 id 去 store 批次查既有內容,內容
    相同的切片直接跳過,只有新增或變更的切片才往下走 embedding → 寫入。

    比對看 ``content`` 與**實際被 embed 的欄位**(``source_field``,
    預設就是 content;``extra_vectors`` 的各來源欄位也算):任一變了就
    放行。store 端這些欄位可能被 indexing 的 ``fields`` 改名後才寫入,
    所以每組比對都帶 (來源欄位, store 端欄位) 名字對。meta 其他欄位
    變了但這些都沒變的切片不會重寫,屬可接受的取捨。來源檔案**刪除**後
    的舊切片不在此元件的守備範圍(它只看這次進來的切片),與全量 ingest
    的 upsert 語意一致:要乾淨索引請刪索引重建。
    """

    def __init__(
        self,
        store: Any,
        source_field: str = "content",
        stored_field: str = "content",
        extra_fields: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        Args:
            store: 目標 document store。
            source_field: 主向量的輸入欄位(切片端名字)。
            stored_field: 主向量輸入欄位在 store 端的名字(可能被改名)。
            extra_fields: extra_vectors 各來源欄位的 (切片端, store 端)
                名字對;這些欄位任一變更也要放行重 embed。
        """
        self.store = store
        self.source_field = source_field
        self.stored_field = stored_field
        self.extra_fields = list(extra_fields or [])

    def _embed_value(self, doc: Document, field: str) -> Any:
        if field == "content":
            return doc.content or ""
        return doc.meta.get(field)

    @component.output_types(documents=list[Document], skipped=int)
    def run(self, documents: list[Document]) -> dict[str, Any]:
        if not documents:
            return {"documents": [], "skipped": 0}
        pairs = [(self.source_field, self.stored_field), *self.extra_fields]
        existing = {
            doc.id: (
                doc.content or "",
                *(self._embed_value(doc, stored) for _, stored in pairs),
            )
            for doc in self.store.filter_documents(
                filters={
                    "field": "id",
                    "operator": "in",
                    "value": [doc.id for doc in documents],
                }
            )
        }
        changed: list[Document] = []
        for doc in documents:
            current = [self._embed_value(doc, source) for source, _ in pairs]
            # 缺值一律放行:讓下游 embedder 的 fail-fast 出聲,
            # 而不是兩邊都 None 相等而被無聲跳過。
            if any(value is None for value in current):
                changed.append(doc)
                continue
            if existing.get(doc.id) != (doc.content or "", *current):
                changed.append(doc)
        skipped = len(documents) - len(changed)
        if skipped:
            logger.info(
                "增量 ingest:跳過 %d/%d 筆未變更的切片(不重算 embedding)",
                skipped, len(documents),
            )
        return {"documents": changed, "skipped": skipped}


@component
class FieldSourceEmbedder:
    """讓任一文件端 embedder 改用指定欄位當輸入,並可加出額外向量。

    主向量:把 ``meta[source_field]`` 暫代 content 交給內部 embedder,
    embed 完把向量抄回**原** Document(content 與 meta 都不變)。
    額外向量(``extra_vectors``,{向量欄位名: 來源欄位},共用同一個
    embedder):逐組以來源欄位再 embed 一輪,結果寫進 ``meta[向量欄位名]``,
    隨 meta 落入索引。任何來源欄位缺值(None / 空字串 / 非字串)直接
    報錯 —— 無聲 fallback 會讓向量空間混入兩種語意。
    """

    def __init__(
        self,
        inner: Any,
        source_field: str,
        extra_vectors: dict[str, str] | None = None,
    ) -> None:
        self.inner = inner
        self.source_field = source_field
        self.extra_vectors = dict(extra_vectors or {})

    def warm_up(self) -> None:
        # Pipeline 只對圖上節點呼叫 warm_up;inner 不在圖上,由這裡委派
        # (sentence_transformers 的 embedder 需要 warm_up 載入模型)。
        if hasattr(self.inner, "warm_up"):
            self.inner.warm_up()

    def _embed_from(
        self, documents: list[Document], field: str, param_desc: str
    ) -> list[Any]:
        """以指定來源欄位跑一輪內部 embedder,回傳向量清單(順序對應輸入)。"""
        if field != "content":
            missing = [
                doc.meta.get("chunk_id") or doc.id
                for doc in documents
                if not isinstance(doc.meta.get(field), str)
                or not doc.meta[field].strip()
            ]
            if missing:
                listed = ", ".join(str(item) for item in missing[:5])
                more = f" 等共 {len(missing)} 筆" if len(missing) > 5 else ""
                raise ComponentError(
                    f"embedding 的 {param_desc} 在以下切片"
                    f"缺值或不是非空字串:{listed}{more}。"
                    "請確認 chunking 元件對每個切片都生成該欄位"
                )
        # content 也走影子複本:有些 embedder 會就地改動傳入的 Document。
        shadows = [
            dataclasses.replace(
                doc, content=doc.content if field == "content" else doc.meta[field]
            )
            for doc in documents
        ]
        embedded = self.inner.run(documents=shadows)["documents"]
        return [doc.embedding for doc in embedded]

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        if not documents:
            return {"documents": []}
        main = self._embed_from(
            documents, self.source_field, f"source_field '{self.source_field}'"
        )
        extras = {
            vec_name: self._embed_from(
                documents, src, f"extra_vectors 來源欄位 '{src}'"
            )
            for vec_name, src in self.extra_vectors.items()
        }
        return {
            "documents": [
                dataclasses.replace(
                    doc,
                    embedding=main[i],
                    meta={
                        **doc.meta,
                        **{name: vectors[i] for name, vectors in extras.items()},
                    },
                )
                for i, doc in enumerate(documents)
            ]
        }


@component
class MetaFieldMapper:
    """依 ``fields``(ES 欄位名 → meta 欄位名)重整 meta(白名單 + 改名)。

    放在 embedder 之後、writer 之前。框架欄位(doc_id / chunk_id / seq /
    page)與 ``preserve`` 列出的欄位(如 extra_vectors 的向量欄位,由框架
    生成、不屬 chunking 的自訂欄位)永遠原名保留;其餘 meta 只保留映射中
    列出者,並以 ES 欄位名寫出。meta 缺某個映射欄位時略過(不塞 None,
    免得污染 ES mapping)。content 與 embedding 是 Document 本體,不受影響。
    """

    FRAMEWORK_FIELDS = ("doc_id", "chunk_id", "seq", "page")

    def __init__(
        self, fields: dict[str, str], preserve: tuple[str, ...] = ()
    ) -> None:
        self.fields = dict(fields)
        self.preserve = tuple(preserve)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        mapped: list[Document] = []
        for doc in documents:
            meta = {
                key: doc.meta[key]
                for key in (*self.FRAMEWORK_FIELDS, *self.preserve)
                if key in doc.meta
            }
            for es_name, meta_name in self.fields.items():
                if meta_name in doc.meta:
                    meta[es_name] = doc.meta[meta_name]
            mapped.append(dataclasses.replace(doc, meta=meta))
        return {"documents": mapped}

"""Ingestion 方法型錄:import / parsing / chunking / embedding / indexing。

每個方法 = 一個參數 schema(pydantic)+ 一個 build 函式 + 對映表一行;
新增方法時照這個模式加一組即可,組裝邏輯(rag/builder.py)不需改動。
custom 方法的相容性宣告寫在 config 參數裡,由 :func:`parsing_declaration` /
:func:`chunking_requires_pages` 於建構期讀出。
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from haystack.components.preprocessors import (
    DocumentCleaner,
    DocumentSplitter,
    RecursiveDocumentSplitter,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pydantic import Field, field_validator, model_validator

from rag.components.api_clients import (
    FlexibleAPIDocumentEmbedder,
    FlexibleAPITextEmbedder,
)
from rag.components.ingestion_steps import FieldSourceEmbedder, FileLister
from rag.components.mock_embedders import MockDocumentEmbedder, MockTextEmbedder
from rag.components.pdf_ocr import PdfToDocument
from rag.custom import CustomModuleParams, instantiate_custom
from rag.errors import ConfigError, MissingDependencyError
from rag.slots import (
    BaseParams,
    BuildContext,
    SlotFactory,
    SlotGraph,
    validate_params,
)

# local_file 支援的副檔名 → content_type 對映。新增檔案型別時:
# 這裡加一列 + `auto` parsing 的 ParsingGraph 加一條分支。
_EXTENSION_CONTENT_TYPES: dict[str, str] = {
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
}
_LOCAL_FILE_DEFAULT_EXTENSIONS = list(_EXTENSION_CONTENT_TYPES)

# 框架保留的欄位名:前四個由 stamper 蓋章,後三個是 Document 本體。
# chunking 的 provides_fields 與 indexing 的 fields 都不可使用這些名字。
_RESERVED_FIELD_NAMES = frozenset(
    {"doc_id", "seq", "page", "chunk_id", "content", "embedding", "id"}
)


def _local_file_output_type(params: dict[str, Any]) -> str:
    """依 extensions 參數推導 import 輸出的 content_type。

    同質(全 text 或全 pdf)→ 該型別;異質 → ``"mixed"``(此時 parsing
    需用 ``auto`` 分流)。藉此保留建構期檢查:extensions 收窄成單一型別
    時,單型別 parser(plain_text / pdf)仍然合法。
    """
    extensions = [
        ext.lower() for ext in (params.get("extensions") or _LOCAL_FILE_DEFAULT_EXTENSIONS)
    ]
    unknown = sorted(set(extensions) - set(_EXTENSION_CONTENT_TYPES))
    if unknown:
        raise ConfigError(
            f"import 方法 'local_file' 不支援副檔名 {unknown};"
            f"目前支援:{sorted(_EXTENSION_CONTENT_TYPES)}"
        )
    types = {_EXTENSION_CONTENT_TYPES[ext] for ext in extensions}
    return next(iter(types)) if len(types) == 1 else "mixed"


class _FileListerParams(BaseParams):
    input_dir: str = Field(description="要匯入的資料夾路徑")
    extensions: list[str] | None = Field(
        default=None, description="要納入的副檔名(預設依方法而定)"
    )
    recursive: bool = Field(default=True, description="是否遞迴掃描子資料夾")


def _make_file_lister_builder(
    method_name: str, default_extensions: list[str]
) -> Callable[[dict[str, Any], BuildContext], Any]:
    def _build(raw: dict[str, Any], ctx: BuildContext) -> FileLister:
        p = validate_params("import", method_name, _FileListerParams, raw)
        return FileLister(
            input_dir=p.input_dir,
            extensions=p.extensions or default_extensions,
            method_name=method_name,
            recursive=p.recursive,
        )

    return _build


class _PlainTextParams(BaseParams):
    encoding: str = Field(default="utf-8", description="文字檔編碼")


def _build_plain_text(raw: dict[str, Any], ctx: BuildContext) -> Any:
    from haystack.components.converters.txt import TextFileToDocument

    p = validate_params("parsing", "plain_text", _PlainTextParams, raw)
    return TextFileToDocument(encoding=p.encoding)


def _normalize_ocr_mode(value: Any) -> Any:
    """把 YAML 解析出的布林還原成 OCR 模式字串。

    YAML 1.1 把裸寫的 ``off`` / ``on`` / ``no`` / ``yes`` 解析成布林,
    ``ocr: off`` 到這裡會是 ``False`` —— 這是使用者最直覺的寫法,
    直接接受(``False`` → "off"、``True`` → "force"),不必強迫加引號。
    """
    if isinstance(value, bool):
        return "force" if value else "off"
    return value


class _PdfParams(BaseParams):
    ocr: Literal["off", "auto", "force"] = Field(
        default="auto",
        description="OCR 策略:off=純 pypdf;auto=掃描頁(無文字層)才 OCR;"
        "force=全頁 OCR(多欄/表格版面順序亂時用)。需 pip install -e \".[ocr]\"",
    )
    ocr_scale: float = Field(default=2.0, gt=0, description="OCR 前的頁面渲染倍率")

    _normalize_ocr = field_validator("ocr", mode="before")(_normalize_ocr_mode)


def _build_pdf(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("parsing", "pdf", _PdfParams, raw)
    return PdfToDocument(mode=p.ocr, ocr_scale=p.ocr_scale)


class _CleanParams(BaseParams):
    remove_empty_lines: bool = Field(default=True)
    remove_extra_whitespaces: bool = Field(default=True)
    remove_repeated_substrings: bool = Field(default=False)


def _build_clean(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("parsing", "clean", _CleanParams, raw)
    return DocumentCleaner(
        remove_empty_lines=p.remove_empty_lines,
        remove_extra_whitespaces=p.remove_extra_whitespaces,
        remove_repeated_substrings=p.remove_repeated_substrings,
    )


class _AutoParsingParams(BaseParams):
    encoding: str = Field(default="utf-8", description="文字 / Markdown 檔編碼")
    ocr: Literal["off", "auto", "force"] = Field(
        default="auto", description="pdf 分支的 OCR 策略(同 pdf 方法的 ocr 參數)"
    )
    ocr_scale: float = Field(default=2.0, gt=0, description="OCR 前的頁面渲染倍率")

    _normalize_ocr = field_validator("ocr", mode="before")(_normalize_ocr_mode)


def _build_auto(raw: dict[str, Any], ctx: BuildContext) -> SlotGraph:
    """依檔案類型分流:txt/md → 文字 converter、pdf → PyPDF,再合流。

    FileTypeRouter 在 meta 非空時會把來源轉成 ByteStream 並 merge meta,
    因此 FileLister 的 doc_id 隨 ByteStream 流進各 converter,meta 契約
    不受分流影響。txt 與 md 各用一個 TextFileToDocument 實例:converter
    的 sources 輸入不是 variadic,兩個 router socket 不能餵同一個實例。
    """
    from haystack.components.converters.txt import TextFileToDocument
    from haystack.components.joiners import DocumentJoiner
    from haystack.components.routers.file_type_router import FileTypeRouter

    p = validate_params("parsing", "auto", _AutoParsingParams, raw)
    return SlotGraph(
        components={
            # Windows 的 mimetypes 不認得 .md → 必須顯式註冊,
            # 否則 .md 全部落進 unclassified、靜默消失。
            "router": FileTypeRouter(
                mime_types=["text/plain", "text/markdown", "application/pdf"],
                additional_mimetypes={"text/markdown": ".md"},
            ),
            "text": TextFileToDocument(encoding=p.encoding),
            "markdown": TextFileToDocument(encoding=p.encoding),
            "pdf": PdfToDocument(mode=p.ocr, ocr_scale=p.ocr_scale),
            # sort_by_score=False:parsing 階段全是 score=None,保持到達
            # 順序(文件內順序是 seq 穩定性的依據)且不觸發 joiner 警告。
            "join": DocumentJoiner(join_mode="concatenate", sort_by_score=False),
        },
        connections=[
            ("router.text/plain", "text.sources"),
            ("router.text/markdown", "markdown.sources"),
            ("router.application/pdf", "pdf.sources"),
            ("text.documents", "join.documents"),
            ("markdown.documents", "join.documents"),
            ("pdf.documents", "join.documents"),
        ],
        inputs={
            "sources": [("router", "sources")],
            "meta": [("router", "meta")],
        },
        output="join",
    )


class _FixedSizeParams(BaseParams):
    split_length: int = Field(default=512, gt=0, description="每個切片的字元數上限")
    split_overlap: int = Field(default=64, ge=0, description="相鄰切片重疊的字元數")

    @model_validator(mode="after")
    def _overlap_lt_length(self) -> "_FixedSizeParams":
        if self.split_overlap >= self.split_length:
            raise ValueError("split_overlap 必須小於 split_length")
        return self


def _build_fixed_size(raw: dict[str, Any], ctx: BuildContext) -> Any:
    # 以字元為單位(v1 語意):中文沒有空白,word 模式幾乎不會切。
    # separators 必須顯式給:預設值含 "sentence",會要求安裝 nltk。
    # "\f"(頁界)放首位:切片不跨頁,splitter 的頁碼標記才正確。
    p = validate_params("chunking", "fixed_size", _FixedSizeParams, raw)
    return RecursiveDocumentSplitter(
        split_length=p.split_length,
        split_overlap=p.split_overlap,
        split_unit="char",
        separators=["\f", "\n\n", "\n", " "],
    )


class _StructureBasedParams(BaseParams):
    split_length: int = Field(default=512, gt=0, description="每個切片的字元數上限")
    split_overlap: int = Field(default=0, ge=0, description="相鄰切片重疊的字元數")
    separators: list[str] = Field(
        default=["\f", "\n\n", "\n", "。", " "],
        description="遞迴切分的分隔符優先序(頁界 → 段落 → 行 → 句 → 空白)",
    )


def _build_structure_based(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("chunking", "structure_based", _StructureBasedParams, raw)
    return RecursiveDocumentSplitter(
        split_length=p.split_length,
        split_overlap=p.split_overlap,
        split_unit="char",
        separators=p.separators,
    )


class _PageBasedParams(BaseParams):
    pages_per_chunk: int = Field(default=1, gt=0, description="每個切片包含的頁數")


def _build_page_based(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("chunking", "page_based", _PageBasedParams, raw)
    return DocumentSplitter(split_by="page", split_length=p.pages_per_chunk)


class _NoChunkingParams(BaseParams):
    pass


def _build_no_chunking(raw: dict[str, Any], ctx: BuildContext) -> None:
    validate_params("chunking", "no_chunking", _NoChunkingParams, raw)
    return None


class _EmbeddingCommonParams(BaseParams):
    """所有 embedding 方法共用:選哪個(些)欄位當文件端 embedding 輸入。"""

    source_field: str = Field(
        default="content",
        description="文件端 embedding 的輸入欄位:content(預設)或 chunking "
        "生成的任一 meta 欄位;查詢端不受影響(查詢文字進同一個模型)",
    )
    extra_vectors: dict[str, str] | None = Field(
        default=None,
        description="額外向量(共用同一個模型):{向量欄位名: 來源欄位}。"
        "主向量照舊寫進 embedding;每組額外向量以指定名字寫進切片 meta,"
        "隨 meta 落入索引(ES 的 kNN 需在 custom_mapping 宣告同名 "
        "dense_vector 欄位)。查詢端內建檢索只用主向量,額外向量供 "
        "custom retrieval 使用",
    )

    @model_validator(mode="after")
    def _validate_extra_vectors(self) -> "_EmbeddingCommonParams":
        if self.extra_vectors is None:
            return self
        bad_names = sorted(set(self.extra_vectors) & _RESERVED_FIELD_NAMES)
        if bad_names:
            raise ValueError(
                f"extra_vectors 的向量欄位名不可使用框架保留名 {bad_names}"
                "(主向量固定寫在 embedding,不需列出)"
            )
        sources = list(self.extra_vectors.values())
        if len(set(sources)) != len(sources):
            raise ValueError(
                "extra_vectors 的來源欄位不可重複"
                "(共用同一個模型時,同一來源只會得到相同向量)"
            )
        collisions = sorted(
            set(self.extra_vectors) & ({self.source_field} | set(sources))
        )
        if collisions:
            raise ValueError(
                f"extra_vectors 的向量欄位名 {collisions} 與來源欄位重名,"
                "向量會覆蓋 meta 中的原文欄位"
            )
        if any(not k.strip() for k in self.extra_vectors) or any(
            not v.strip() for v in sources
        ):
            raise ValueError("extra_vectors 的鍵與值不可為空字串")
        return self


def _wrap_source_field(
    doc_embedder: Any,
    source_field: str,
    extra_vectors: dict[str, str] | None = None,
) -> Any:
    """輸入來源非預設時把文件端 embedder 包進 FieldSourceEmbedder。

    只在非預設時包:元件名照舊是 "embedder",預設路徑的圖形狀完全不變。
    """
    if source_field == "content" and not extra_vectors:
        return doc_embedder
    return FieldSourceEmbedder(doc_embedder, source_field, extra_vectors)


class _MockEmbeddingParams(_EmbeddingCommonParams):
    dim: int = Field(default=32, gt=1, description="向量維度")


def _build_mock_embedding(raw: dict[str, Any], ctx: BuildContext) -> tuple[Any, Any]:
    p = validate_params("embedding", "mock", _MockEmbeddingParams, raw)
    return (
        _wrap_source_field(
            MockDocumentEmbedder(dim=p.dim), p.source_field, p.extra_vectors
        ),
        MockTextEmbedder(dim=p.dim),
    )


class _ApiEmbeddingParams(_EmbeddingCommonParams):
    endpoint: str = Field(description="embedding API 端點 URL")
    headers: dict[str, str] = Field(default_factory=dict)
    model: str | None = Field(default=None, description="模型名稱(None 時請求不帶此欄位)")
    batch_size: int = Field(default=16, gt=0)
    timeout: float = Field(default=30.0, gt=0)
    texts_field: str = Field(default="input")
    model_field: str = Field(default="model")
    embeddings_field: str | None = Field(default="embeddings")
    item_field: str | None = Field(default=None)


def _build_api_embedding(raw: dict[str, Any], ctx: BuildContext) -> tuple[Any, Any]:
    p = validate_params("embedding", "api_embedding", _ApiEmbeddingParams, raw)
    kwargs = p.model_dump()
    # 框架參數,不透傳給 API 元件的建構子
    kwargs.pop("source_field")
    kwargs.pop("extra_vectors")
    return (
        _wrap_source_field(
            FlexibleAPIDocumentEmbedder(**kwargs), p.source_field, p.extra_vectors
        ),
        FlexibleAPITextEmbedder(**kwargs),
    )


class _SentenceTransformersParams(_EmbeddingCommonParams):
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", description="模型名稱"
    )


def _build_sentence_transformers(
    raw: dict[str, Any], ctx: BuildContext
) -> tuple[Any, Any]:
    p = validate_params(
        "embedding", "sentence_transformers", _SentenceTransformersParams, raw
    )
    try:
        from haystack_integrations.components.embedders.sentence_transformers import (
            SentenceTransformersDocumentEmbedder,
            SentenceTransformersTextEmbedder,
        )
    except ImportError as exc:
        raise MissingDependencyError(
            "sentence-transformers-haystack",
            "sentence_transformers embedding 方法",
        ) from exc
    return (
        _wrap_source_field(
            SentenceTransformersDocumentEmbedder(model=p.model_name),
            p.source_field,
            p.extra_vectors,
        ),
        SentenceTransformersTextEmbedder(model=p.model_name),
    )


class _IndexingCommonParams(BaseParams):
    incremental: bool = Field(
        default=False,
        description="增量 ingest:內容未變的切片跳過 embedding 與寫入"
        "(需要索引具備 incremental_update 能力)",
    )
    fields: dict[str, str] | None = Field(
        default=None,
        description="寫入索引的自訂欄位白名單 + 改名:{索引欄位名: meta 欄位名}。"
        "設定後未列出的自訂 meta 欄位不寫入;框架欄位"
        "(doc_id/chunk_id/seq/page)與 content/embedding 永遠保留",
    )

    @model_validator(mode="after")
    def _validate_fields(self) -> "_IndexingCommonParams":
        if self.fields is None:
            return self
        bad_keys = sorted(set(self.fields) & _RESERVED_FIELD_NAMES)
        if bad_keys:
            raise ValueError(
                f"fields 的索引欄位名不可使用框架保留名 {bad_keys}"
                "(這些欄位由框架寫入)"
            )
        bad_values = sorted(set(self.fields.values()) & _RESERVED_FIELD_NAMES)
        if bad_values:
            raise ValueError(
                f"fields 引用的 meta 欄位 {bad_values} 是框架欄位,"
                "本來就會寫入,不需列在 fields(fields 只作用於自訂欄位)"
            )
        values = list(self.fields.values())
        if len(set(values)) != len(values):
            raise ValueError(
                "fields 的 meta 欄位名不可重複(同一個欄位映到多個索引欄位名)"
            )
        if any(not k.strip() for k in self.fields) or any(
            not v.strip() for v in values
        ):
            raise ValueError("fields 的鍵與值不可為空字串")
        return self


class _InMemoryParams(_IndexingCommonParams):
    pass


def _build_in_memory_store(raw: dict[str, Any], ctx: BuildContext) -> Any:
    validate_params("indexing", "in_memory", _InMemoryParams, raw)
    # BM25 斷詞:預設的 \w+ 會把整段中文吞成一個 token,中文查詢幾乎
    # 比不到;改為「CJK 逐字 + 拉丁/數字詞」(ES 的 standard analyzer
    # 對 CJK 也是逐字,行為一致)。
    return InMemoryDocumentStore(
        embedding_similarity_function="cosine",
        bm25_tokenization_regex=(
            r"[㐀-䶿一-鿿豈-﫿]|[A-Za-z0-9_]+"
        ),
    )


class _ElasticsearchParams(_IndexingCommonParams):
    hosts: str | list[str] = Field(description="ES 端點,如 http://localhost:9200")
    index: str = Field(default="modular-rag", description="索引名稱")
    api_key: str | None = Field(
        default=None,
        description="ES API key(base64 的 'id:api_key' 形式);與 username/password 擇一",
    )
    username: str | None = Field(
        default=None, description="basic auth 帳號;需與 password 成對提供"
    )
    password: str | None = Field(
        default=None, description="basic auth 密碼;需與 username 成對提供"
    )
    ca_certs: str | None = Field(
        default=None, description="CA 憑證路徑(https 叢集用私有 CA 簽發時需要)"
    )
    verify_certs: bool | None = Field(
        default=None, description="是否驗證伺服器憑證(預設 true;關閉會失去傳輸層保護)"
    )
    custom_mapping: dict[str, Any] | None = Field(
        default=None,
        description="完整覆蓋預設索引 mapping(欄位可指定 plugin analyzer;"
        "須自行涵蓋 content 與 embedding dense_vector 欄位,dims 須等於"
        "embedding 模型維度)。僅在索引不存在、由框架建立時生效;"
        "需要 analyzer 定義(settings.analysis)時搭配 settings 參數",
    )
    settings: dict[str, Any] | None = Field(
        default=None,
        description="建索引時的 ES index settings(analysis / shards 等)。"
        "Haystack 原生只吃 mapping,設定本參數時由框架在索引不存在時預建"
        "(需同時提供 custom_mapping);注意建 pipeline 時就會連線 ES",
    )
    ingest_pipeline: str | None = Field(
        default=None,
        description="寫入文件時套用的 ES ingest pipeline 名稱(伺服器端須已建立)",
    )
    request_timeout: float | None = Field(
        default=None,
        gt=0,
        description="單次 ES 請求的逾時秒數(client 預設 10),對寫入用的 "
        "store 與 settings 預建索引的連線都生效。整批 bulk 寫入"
        "在資料量大或線路慢時會超過 10 秒 —— 症狀是 writer 步驟拋 "
        "'Connection timed out',log 中該次 _bulk 請求是 status:N/A、"
        "duration 恰好等於逾時值。注意:若索引的 refresh_interval 設得很長"
        "(公司 index template 常見 30s / -1),寫入預設會等下一次 refresh,"
        "此時調高本參數只是讓它空等 —— 病因在 refresh_interval",
    )
    retry_on_timeout: bool | None = Field(
        default=None,
        description="請求逾時是否自動重試(client 預設 false)。線路間歇不穩"
        "(如 settings 預建索引的 HEAD 偶發 timeout)時設 true;"
        "連線層錯誤 client 本來就會重試,本參數只擴及逾時",
    )
    max_retries: int | None = Field(
        default=None,
        ge=1,
        description="單次請求的最大重試次數(client 預設 3);"
        "搭配 retry_on_timeout 調整逾時重試的上限",
    )

    @model_validator(mode="after")
    def _settings_require_mapping(self) -> "_ElasticsearchParams":
        if self.settings is not None and self.custom_mapping is None:
            raise ValueError(
                "settings 需要搭配 custom_mapping:框架會用兩者預建索引,"
                "只給 settings 會讓 content/embedding 欄位落入 dynamic mapping"
                "(dense_vector 型別遺失)"
            )
        return self


def _elasticsearch_auth_kwargs(p: Any) -> dict[str, Any]:
    """把認證欄位翻成 elasticsearch client 參數,並擋掉半套的組合。

    開了 security 的叢集(公司 / Elastic Cloud 叢集預設如此)沒帶憑證時,
    第一個請求就會收到 401 ``missing authentication credentials``;
    本函式在建 store 前先確認 config 給的認證資訊是完整且不衝突的。
    """
    if (p.username is None) != (p.password is None):
        missing = "password" if p.username is not None else "username"
        raise ConfigError(
            f"模組 'indexing' 方法 'elasticsearch' 的 username 與 password 必須成對提供,"
            f"目前缺少 {missing}。"
        )
    if p.api_key is not None and p.username is not None:
        raise ConfigError(
            "模組 'indexing' 方法 'elasticsearch' 的 api_key 與 username/password 只能擇一:"
            "Elasticsearch client 不接受同時帶兩種認證。"
        )
    if p.api_key is not None:
        return {"api_key": p.api_key}
    if p.username is not None:
        # 必須是 2 元組/清單:字串形式會被 client 當成「已編碼」原樣送出。
        return {"basic_auth": (p.username, p.password)}
    return {}


def _elasticsearch_client_kwargs(p: _ElasticsearchParams) -> dict[str, Any]:
    """兩個連線點共用的 elasticsearch client 參數(不設定就不帶)。

    寫入用的 store 與 settings 預建索引的 client 都吃這組:認證、TLS 與
    timeout / 重試行為必須一致,否則預建路徑會用 client 預設(10 秒、
    不重試)打第一個請求,調了 request_timeout 也治不到它。
    """
    kwargs: dict[str, Any] = {**_elasticsearch_auth_kwargs(p)}
    if p.ca_certs is not None:
        kwargs["ca_certs"] = p.ca_certs
    if p.verify_certs is not None:
        kwargs["verify_certs"] = p.verify_certs
    if p.request_timeout is not None:
        kwargs["request_timeout"] = p.request_timeout
    if p.retry_on_timeout is not None:
        kwargs["retry_on_timeout"] = p.retry_on_timeout
    if p.max_retries is not None:
        kwargs["max_retries"] = p.max_retries
    return kwargs


def _elasticsearch_store_kwargs(p: _ElasticsearchParams) -> dict[str, Any]:
    """組出 ElasticsearchDocumentStore 的建構參數(選填欄位不設定就不帶)。

    client 參數靠 store 的 **kwargs 原樣轉給 Elasticsearch(...);
    request_timeout / retry_on_timeout / max_retries 都走這條路。
    """
    kwargs: dict[str, Any] = {
        "hosts": p.hosts,
        "index": p.index,
        "embedding_similarity_function": "cosine",
        **_elasticsearch_client_kwargs(p),
    }
    if p.custom_mapping is not None:
        kwargs["custom_mapping"] = p.custom_mapping
    if p.ingest_pipeline is not None:
        kwargs["ingest_pipeline"] = p.ingest_pipeline
    return kwargs


def _ensure_es_index(client: Any, p: _ElasticsearchParams) -> None:
    """索引不存在時以 custom_mapping + settings 預建(存在則不動)。

    Haystack 的 store 建索引只吃 mapping,settings(analysis 等)進不去;
    但它看到索引已存在會跳過自建 —— 所以框架先建,store 接手用。
    """
    if client.indices.exists(index=p.index):
        return
    client.indices.create(
        index=p.index, mappings=p.custom_mapping, settings=p.settings
    )


def _build_elasticsearch_store(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("indexing", "elasticsearch", _ElasticsearchParams, raw)
    kwargs = _elasticsearch_store_kwargs(p)
    try:
        from haystack_integrations.document_stores.elasticsearch import (
            ElasticsearchDocumentStore,
        )
    except ImportError as exc:
        raise MissingDependencyError(
            "elasticsearch-haystack", "elasticsearch indexing 方法"
        ) from exc
    if p.settings is not None:
        # 注意:僅此路徑在建 pipeline 時就連線 ES(預建需要檢查索引存在)。
        from elasticsearch import Elasticsearch

        client = Elasticsearch(p.hosts, **_elasticsearch_client_kwargs(p))
        _ensure_es_index(client, p)
    return ElasticsearchDocumentStore(**kwargs)


class _CustomImportParams(CustomModuleParams):
    """``import: method: custom`` 的參數:custom module 定位 + content_type 宣告。"""

    content_type: Literal["text", "pdf", "mixed"] | None = Field(
        default=None,
        description="元件輸出的 content_type 宣告,供建構期與 parsing 的相容性"
        "檢查;None = 跳過檢查(元件輸出的型態由你自行保證)",
    )


def _build_custom_import(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("import", "custom", _CustomImportParams, raw)
    return instantiate_custom("import", p)


def _custom_import_output_type(raw: dict[str, Any]) -> str | None:
    p = validate_params("import", "custom", _CustomImportParams, raw)
    return p.content_type


IMPORT_FACTORIES: dict[str, SlotFactory] = {
    # 萬用 importer:txt / md / pdf 都收(可用 extensions 收窄);
    # content_type 依 extensions 推導,混合型別時 parsing 需用 auto 分流。
    "local_file": SlotFactory(
        build=_make_file_lister_builder("local_file", _LOCAL_FILE_DEFAULT_EXTENSIONS),
        output_content_type_fn=_local_file_output_type,
    ),
    # 契約:無輸入 → sources + meta(meta 每筆必帶 doc_id)。
    # sources 可以是路徑或 ByteStream(公司 API 直接回內容時);
    # 非路徑 sources 搭配 incremental 時檔案層增量會退化為全量重 parse。
    "custom": SlotFactory(
        build=_build_custom_import,
        output_content_type_fn=_custom_import_output_type,
    ),
}

# parsing 的 content_type 全集:custom converter 未宣告 input_content_types
# 時的預設(封閉 Literal 之下,「全收」與「跳過檢查」同義)。
_ALL_CONTENT_TYPES = frozenset(set(_EXTENSION_CONTENT_TYPES.values()) | {"mixed"})


class _CustomParsingParams(CustomModuleParams):
    """``parsing: method: custom`` 的參數:custom module 定位 + 鏈位置宣告。"""

    kind: Literal["converter", "doc_processor"] = Field(
        default="doc_processor",
        description="鏈位置:converter = 鏈首(sources + meta → documents);"
        "doc_processor = 鏈中 / 鏈尾(documents → documents,預設)",
    )
    produces_pages: bool = Field(
        default=False,
        description="元件是否產生頁界資訊(page_based chunking 的前提;"
        "兩種 kind 都可宣告)",
    )
    input_content_types: list[Literal["text", "pdf", "mixed"]] | None = Field(
        default=None,
        description="converter 可處理的 content_type(供與 import 的相容性檢查);"
        "None = 全收。僅 kind: converter 可設定",
    )

    @model_validator(mode="after")
    def _content_types_only_for_converter(self) -> "_CustomParsingParams":
        if self.kind != "converter" and self.input_content_types is not None:
            raise ValueError(
                "input_content_types 只對 kind: converter 有意義"
                "(doc_processor 收的是上游的 documents,不接觸原始檔案)"
            )
        return self


def _build_custom_parsing(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("parsing", "custom", _CustomParsingParams, raw)
    contract = "parsing_converter" if p.kind == "converter" else "parsing"
    return instantiate_custom("parsing", p, contract=contract)


def parsing_declaration(
    method: str, factory: SlotFactory, params: dict[str, Any]
) -> tuple[str, bool, frozenset[str]]:
    """回傳 parsing 方法的 ``(kind, produces_pages, input_content_types)``。

    內建方法讀 factory 的靜態宣告;custom 的宣告寫在 config 參數裡
    (kind / produces_pages / input_content_types),從 params 讀出。
    """
    if method != "custom":
        return factory.kind, factory.produces_pages, factory.input_content_types
    p = validate_params("parsing", "custom", _CustomParsingParams, params)
    input_types = (
        _ALL_CONTENT_TYPES
        if p.input_content_types is None
        else frozenset(p.input_content_types)
    )
    return p.kind, p.produces_pages, input_types


PARSING_FACTORIES: dict[str, SlotFactory] = {
    "plain_text": SlotFactory(
        build=_build_plain_text,
        kind="converter",
        input_content_types=frozenset({"text"}),
    ),
    "pdf": SlotFactory(
        build=_build_pdf,
        kind="converter",
        input_content_types=frozenset({"pdf"}),
        produces_pages=True,
    ),
    # 依檔案類型分流(混合 KB)。produces_pages=True:pdf 分支產生頁界;
    # txt/md 在 page_based 下視為單頁(與「非分頁來源 page=1」語意一致)。
    "auto": SlotFactory(
        build=_build_auto,
        kind="converter",
        input_content_types=frozenset({"text", "pdf", "mixed"}),
        produces_pages=True,
    ),
    "clean": SlotFactory(build=_build_clean, kind="doc_processor"),
    # 契約依 kind 而異:converter(鏈首)= sources + meta → documents;
    # doc_processor(鏈中,預設)= documents → documents。相容性宣告
    # 寫在 config 參數裡,由 parsing_declaration 讀出。
    "custom": SlotFactory(build=_build_custom_parsing),
}


class _CustomChunkingParams(CustomModuleParams):
    """``chunking: method: custom`` 的參數:custom module 定位 + 宣告欄位。"""

    requires_pages: bool = Field(
        default=False,
        description="元件是否需要分頁輸入(供與 parsing 鏈的相容性檢查:"
        "true 時 parsing 鏈必須產生頁界)",
    )
    provides_fields: list[str] | None = Field(
        default=None,
        description="元件會在切片 meta 生成的欄位(選配):宣告後,"
        "embedding 的 source_field 與 indexing 的 fields 引用未宣告欄位時"
        "建構期即報錯;未宣告則退化為執行期檢查",
    )

    @field_validator("provides_fields")
    @classmethod
    def _no_reserved_or_duplicate_fields(
        cls, value: list[str] | None
    ) -> list[str] | None:
        if value is None:
            return value
        reserved = sorted(set(value) & _RESERVED_FIELD_NAMES)
        if reserved:
            raise ValueError(
                f"provides_fields 不可包含框架保留欄位 {reserved}"
                "(doc_id/seq/page/chunk_id 由框架蓋章,content/embedding/id "
                "是 Document 本體)"
            )
        if len(set(value)) != len(value) or any(not name.strip() for name in value):
            raise ValueError("provides_fields 不可有重複或空白的欄位名")
        return value


def _build_custom_chunking(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("chunking", "custom", _CustomChunkingParams, raw)
    return instantiate_custom("chunking", p)


def chunking_requires_pages(
    method: str, factory: SlotFactory, params: dict[str, Any]
) -> bool:
    """回傳 chunking 方法是否需要分頁輸入(custom 的宣告寫在 config 參數裡)。"""
    if method != "custom":
        return factory.requires_pages
    return validate_params(
        "chunking", "custom", _CustomChunkingParams, params
    ).requires_pages


def chunking_provides_fields(
    method: str, factory: SlotFactory, params: dict[str, Any]
) -> list[str] | None:
    """回傳 chunking custom 宣告的生成欄位;未宣告或非 custom 回 None。"""
    if method != "custom":
        return None
    return validate_params(
        "chunking", "custom", _CustomChunkingParams, params
    ).provides_fields


CHUNKING_FACTORIES: dict[str, SlotFactory] = {
    "fixed_size": SlotFactory(build=_build_fixed_size),
    "structure_based": SlotFactory(build=_build_structure_based),
    "page_based": SlotFactory(build=_build_page_based, requires_pages=True),
    "no_chunking": SlotFactory(build=_build_no_chunking),
    # 契約:documents → documents(meta 逐塊繼承,doc_id 必須保留 ——
    # 下游 stamper 依 doc_id 產生穩定 chunk_id)。分頁需求宣告寫在
    # config 參數裡,由 chunking_requires_pages 讀出。
    "custom": SlotFactory(build=_build_custom_chunking),
}

EMBEDDING_FACTORIES: dict[str, SlotFactory] = {
    "mock": SlotFactory(build=_build_mock_embedding),
    "api_embedding": SlotFactory(build=_build_api_embedding),
    "sentence_transformers": SlotFactory(build=_build_sentence_transformers),
}

INDEXING_FACTORIES: dict[str, SlotFactory] = {
    # in_memory 也宣告 incremental_update:store 生命週期內按 id 查找 /
    # upsert 都支援(僅不跨重啟,但那是持久性問題,不是能力問題)。
    "in_memory": SlotFactory(
        build=_build_in_memory_store,
        capabilities=frozenset(
            {"vector_search", "text_search", "metadata_filter", "incremental_update"}
        ),
    ),
    "elasticsearch": SlotFactory(
        build=_build_elasticsearch_store,
        capabilities=frozenset(
            {"vector_search", "text_search", "metadata_filter", "incremental_update"}
        ),
    ),
}



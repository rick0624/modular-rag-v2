"""薄 builder:把槽位式 config 翻譯成 Haystack Pipeline。

這是整個框架的核心(也是唯一)自維護層:每個槽位一張
「方法名稱 → :class:`SlotFactory`」對映表,factory 負責驗證參數並
建立對應的 Haystack 元件;圖的接線由 builder 統一處理,config 只需
選方法、填參數。語意層的相容性檢查(content_type / requires_pages /
索引能力)在建構期執行,不合法的組合直接報錯並列出可用替代。

新增一個方法 = 寫一個 factory(或自訂元件)+ 在對映表加一行;
不需要改動任何其他程式碼。
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable, Literal

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import (
    DocumentCleaner,
    DocumentSplitter,
    RecursiveDocumentSplitter,
)
from haystack.components.rankers import LLMRanker
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from rag.compatibility import (
    validate_chunking_compatibility,
    validate_inference_compatibility,
    validate_ingestion_compatibility,
)
from rag.components.api_embedders import (
    FlexibleAPIDocumentEmbedder,
    FlexibleAPITextEmbedder,
)
from rag.components.api_reranker import FlexibleAPIRanker
from rag.components.change_filter import IncrementalChangeFilter
from rag.components.file_lister import FileLister
from rag.components.pdf_ocr import PdfToDocument
from rag.components.source_filter import SourceChangeFilter
from rag.components.fusion import SubqueryFusion
from rag.components.gateway_generator import GatewayChatGenerator, MockChatGenerator
from rag.components.meta_stamper import ChunkMetaStamper
from rag.components.mock_embedders import MockDocumentEmbedder, MockTextEmbedder
from rag.components.multi_query import MultiQueryRetrievalStage
from rag.components.fact_check import DEFAULT_FACT_CHECK_PROMPT, LLMFactChecker
from rag.components.query_transforms import (
    DEFAULT_DECOMPOSE_PROMPT,
    DEFAULT_REWRITE_PROMPT,
    GlossaryExpander,
    JargonMapper,
    LLMQueryDecomposer,
    LLMQueryRewriter,
    QueryNormalizer,
)
from rag.components.routing import KeywordRouteClassifier
from rag.config import FusionConfig, MethodConfig, RAGConfig
from rag.custom import CustomModuleParams, instantiate_custom
from rag.errors import (
    ConfigError,
    IncompatiblePipelineError,
    MissingDependencyError,
    UnknownMethodError,
)
from rag.trace import step_order

logger = logging.getLogger(__name__)

# 輸入輸出同型別、支援方法鏈(method 清單)的槽位。
CHAINABLE_SLOTS = frozenset({"parsing", "query_transformation", "reranking"})

# build_pipelines(stage=…) 可組的階段。順序即 CLI 的顯示順序。
PIPELINE_STAGES: tuple[str, ...] = ("all", "ingestion", "inference")


# ---------------------------------------------------------------------------
# 基礎設施:SlotFactory / BuildContext / 參數驗證
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotFactory:
    """一個方法的建立函式 + 宣告式相容性欄位。

    ``build(params, ctx)`` 回傳該方法對應的 Haystack 元件
    (embedding 槽位回傳 ``(document_embedder, text_embedder)`` 一對,
    indexing 槽位回傳 document store,``no_chunking`` 等可回傳 None)。
    """

    build: Callable[[dict[str, Any], "BuildContext"], Any]
    kind: str = ""  # parsing 鏈用:"converter"(檔案→Document)/ "doc_processor"
    output_content_type: str | None = None  # import 槽位宣告(靜態)
    output_content_type_fn: Callable[[dict[str, Any]], str | None] | None = None
    """import 槽位宣告(動態):依方法參數推導 content_type(如 local_file
    依 extensions 推導 text / pdf / mixed);設定時優先於靜態宣告。"""
    input_content_types: frozenset[str] = frozenset()  # parsing 槽位宣告
    produces_pages: bool = False  # parsing 槽位宣告
    requires_pages: bool = False  # chunking 槽位宣告
    capabilities: frozenset[str] = frozenset()  # indexing 槽位宣告
    required_capabilities: frozenset[str] = frozenset()  # retrieval 槽位宣告

    # 動態宣告(custom 方法用):相容性欄位寫在 config 參數裡,建構期以
    # :func:`_resolved_factory` 解析成靜態欄位的複本 —— 鏈驗證與
    # compatibility 檢查因此完全不需要認識 custom。與 output_content_type_fn
    # 同一先例,設定時優先於對應的靜態欄位。
    kind_fn: Callable[[dict[str, Any]], str] | None = None
    produces_pages_fn: Callable[[dict[str, Any]], bool] | None = None
    input_content_types_fn: Callable[[dict[str, Any]], frozenset[str]] | None = None
    requires_pages_fn: Callable[[dict[str, Any]], bool] | None = None


def _resolved_factory(factory: SlotFactory, params: dict[str, Any]) -> SlotFactory:
    """把 factory 的動態宣告(``*_fn``)依方法參數解析成靜態欄位複本。

    非 custom 方法沒有設定任何 ``*_fn``,原樣返回(零成本)。
    """
    updates: dict[str, Any] = {}
    if factory.kind_fn is not None:
        updates["kind"] = factory.kind_fn(params)
    if factory.produces_pages_fn is not None:
        updates["produces_pages"] = factory.produces_pages_fn(params)
    if factory.input_content_types_fn is not None:
        updates["input_content_types"] = factory.input_content_types_fn(params)
    if factory.requires_pages_fn is not None:
        updates["requires_pages"] = factory.requires_pages_fn(params)
    return dataclasses.replace(factory, **updates) if updates else factory


@dataclass
class BuildContext:
    """建構期共享的執行環境(store、方法名稱、跨槽位依賴)。"""

    store: Any = None
    indexing_method: str = ""
    embedding_config: MethodConfig | None = None
    generation_config: MethodConfig | None = None


class BaseParams(BaseModel):
    """所有方法參數 schema 的基底:多打欄位直接報錯,不靜默忽略。"""

    model_config = ConfigDict(extra="forbid")


def _validate_params(
    slot: str, method: str, params_cls: type[BaseParams], raw: dict[str, Any]
) -> Any:
    """以方法的 Params schema 驗證參數,錯誤時列出可接受的參數。"""
    try:
        return params_cls(**raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        accepted = ", ".join(sorted(params_cls.model_fields.keys())) or "(不接受任何參數)"
        raise ConfigError(
            f"模組 '{slot}' 方法 '{method}' 的參數不合法:{details}。"
            f"可接受的參數:{accepted}"
        ) from exc


def _resolve(
    slot: str, table: dict[str, SlotFactory], method: str
) -> SlotFactory:
    """從對映表取出方法的 factory;不存在時列出所有可用方法。"""
    if method not in table:
        raise UnknownMethodError(slot, method, list(table.keys()))
    return table[method]


def _require_single(slot: str, cfg: MethodConfig) -> str:
    """取出唯一的方法名稱;不支援鏈的槽位收到清單時報錯。"""
    methods = cfg.methods()
    if len(methods) > 1 and slot not in CHAINABLE_SLOTS:
        raise ConfigError(
            f"模組 '{slot}' 的輸入輸出型別不同,不支援方法鏈;"
            f"收到 method 清單 {methods},請指定單一方法"
        )
    return methods[0]


# ---------------------------------------------------------------------------
# Ingestion 槽位的 factories
# ---------------------------------------------------------------------------


# local_file 支援的副檔名 → content_type 對映。新增檔案型別時:
# 這裡加一列 + `auto` parsing 的 ParsingGraph 加一條分支。
_EXTENSION_CONTENT_TYPES: dict[str, str] = {
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
}
_LOCAL_FILE_DEFAULT_EXTENSIONS = list(_EXTENSION_CONTENT_TYPES)


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
        p = _validate_params("import", method_name, _FileListerParams, raw)
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

    p = _validate_params("parsing", "plain_text", _PlainTextParams, raw)
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
    p = _validate_params("parsing", "pdf", _PdfParams, raw)
    return PdfToDocument(mode=p.ocr, ocr_scale=p.ocr_scale)


class _CleanParams(BaseParams):
    remove_empty_lines: bool = Field(default=True)
    remove_extra_whitespaces: bool = Field(default=True)
    remove_repeated_substrings: bool = Field(default=False)


def _build_clean(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("parsing", "clean", _CleanParams, raw)
    return DocumentCleaner(
        remove_empty_lines=p.remove_empty_lines,
        remove_extra_whitespaces=p.remove_extra_whitespaces,
        remove_repeated_substrings=p.remove_repeated_substrings,
    )


@dataclass
class ParsingGraph:
    """parsing 鏈首方法展開後的內部圖片段(多分支 converter)。

    仿 :class:`_RetrievalGraph`:factory 回傳圖描述,由
    :func:`build_ingestion_pipeline` 展開接線。對外契約不變 ——
    輸入 ``sources`` + ``meta``、輸出 ``documents``。
    """

    components: dict[str, Any]  # 相對名稱 → 元件
    connections: list[tuple[str, str]]  # 圖內接線("元件.socket" 相對名)
    sources_target: tuple[str, str]  # importer.sources 的落點
    meta_target: tuple[str, str]  # importer.meta 的落點
    output: str  # 輸出 documents 的元件(相對名稱)


class _AutoParsingParams(BaseParams):
    encoding: str = Field(default="utf-8", description="文字 / Markdown 檔編碼")
    ocr: Literal["off", "auto", "force"] = Field(
        default="auto", description="pdf 分支的 OCR 策略(同 pdf 方法的 ocr 參數)"
    )
    ocr_scale: float = Field(default=2.0, gt=0, description="OCR 前的頁面渲染倍率")

    _normalize_ocr = field_validator("ocr", mode="before")(_normalize_ocr_mode)


def _build_auto(raw: dict[str, Any], ctx: BuildContext) -> ParsingGraph:
    """依檔案類型分流:txt/md → 文字 converter、pdf → PyPDF,再合流。

    FileTypeRouter 在 meta 非空時會把來源轉成 ByteStream 並 merge meta,
    因此 FileLister 的 doc_id 隨 ByteStream 流進各 converter,meta 契約
    不受分流影響。txt 與 md 各用一個 TextFileToDocument 實例:converter
    的 sources 輸入不是 variadic,兩個 router socket 不能餵同一個實例。
    """
    from haystack.components.converters.txt import TextFileToDocument
    from haystack.components.joiners import DocumentJoiner
    from haystack.components.routers.file_type_router import FileTypeRouter

    p = _validate_params("parsing", "auto", _AutoParsingParams, raw)
    return ParsingGraph(
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
        sources_target=("router", "sources"),
        meta_target=("router", "meta"),
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
    p = _validate_params("chunking", "fixed_size", _FixedSizeParams, raw)
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
    p = _validate_params("chunking", "structure_based", _StructureBasedParams, raw)
    return RecursiveDocumentSplitter(
        split_length=p.split_length,
        split_overlap=p.split_overlap,
        split_unit="char",
        separators=p.separators,
    )


class _PageBasedParams(BaseParams):
    pages_per_chunk: int = Field(default=1, gt=0, description="每個切片包含的頁數")


def _build_page_based(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("chunking", "page_based", _PageBasedParams, raw)
    return DocumentSplitter(split_by="page", split_length=p.pages_per_chunk)


class _NoChunkingParams(BaseParams):
    pass


def _build_no_chunking(raw: dict[str, Any], ctx: BuildContext) -> None:
    _validate_params("chunking", "no_chunking", _NoChunkingParams, raw)
    return None


class _MockEmbeddingParams(BaseParams):
    dim: int = Field(default=32, gt=1, description="向量維度")


def _build_mock_embedding(raw: dict[str, Any], ctx: BuildContext) -> tuple[Any, Any]:
    p = _validate_params("embedding", "mock", _MockEmbeddingParams, raw)
    return (MockDocumentEmbedder(dim=p.dim), MockTextEmbedder(dim=p.dim))


class _ApiEmbeddingParams(BaseParams):
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
    p = _validate_params("embedding", "api_embedding", _ApiEmbeddingParams, raw)
    kwargs = p.model_dump()
    return (
        FlexibleAPIDocumentEmbedder(**kwargs),
        FlexibleAPITextEmbedder(**kwargs),
    )


class _SentenceTransformersParams(BaseParams):
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", description="模型名稱"
    )


def _build_sentence_transformers(
    raw: dict[str, Any], ctx: BuildContext
) -> tuple[Any, Any]:
    p = _validate_params(
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
        SentenceTransformersDocumentEmbedder(model=p.model_name),
        SentenceTransformersTextEmbedder(model=p.model_name),
    )


class _IndexingCommonParams(BaseParams):
    incremental: bool = Field(
        default=False,
        description="增量 ingest:內容未變的切片跳過 embedding 與寫入"
        "(需要索引具備 incremental_update 能力)",
    )


class _InMemoryParams(_IndexingCommonParams):
    pass


def _build_in_memory_store(raw: dict[str, Any], ctx: BuildContext) -> Any:
    _validate_params("indexing", "in_memory", _InMemoryParams, raw)
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


def _build_elasticsearch_store(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("indexing", "elasticsearch", _ElasticsearchParams, raw)
    auth = _elasticsearch_auth_kwargs(p)
    try:
        from haystack_integrations.document_stores.elasticsearch import (
            ElasticsearchDocumentStore,
        )
    except ImportError as exc:
        raise MissingDependencyError(
            "elasticsearch-haystack", "elasticsearch indexing 方法"
        ) from exc
    kwargs: dict[str, Any] = {
        "hosts": p.hosts,
        "index": p.index,
        "embedding_similarity_function": "cosine",
        **auth,
    }
    if p.ca_certs is not None:
        kwargs["ca_certs"] = p.ca_certs
    if p.verify_certs is not None:
        kwargs["verify_certs"] = p.verify_certs
    return ElasticsearchDocumentStore(**kwargs)


class _CustomImportParams(CustomModuleParams):
    """``import: method: custom`` 的參數:custom module 定位 + content_type 宣告。"""

    content_type: Literal["text", "pdf", "mixed"] | None = Field(
        default=None,
        description="元件輸出的 content_type 宣告,供建構期與 parsing 的相容性"
        "檢查;None = 跳過檢查(元件輸出的型態由你自行保證)",
    )


def _build_custom_import(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("import", "custom", _CustomImportParams, raw)
    return instantiate_custom("import", p)


def _custom_import_output_type(raw: dict[str, Any]) -> str | None:
    p = _validate_params("import", "custom", _CustomImportParams, raw)
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
    p = _validate_params("parsing", "custom", _CustomParsingParams, raw)
    contract = "parsing_converter" if p.kind == "converter" else "parsing"
    return instantiate_custom("parsing", p, contract=contract)


def _custom_parsing_kind(raw: dict[str, Any]) -> str:
    return _validate_params("parsing", "custom", _CustomParsingParams, raw).kind


def _custom_parsing_produces_pages(raw: dict[str, Any]) -> bool:
    return _validate_params(
        "parsing", "custom", _CustomParsingParams, raw
    ).produces_pages


def _custom_parsing_input_types(raw: dict[str, Any]) -> frozenset[str]:
    p = _validate_params("parsing", "custom", _CustomParsingParams, raw)
    if p.input_content_types is None:
        return _ALL_CONTENT_TYPES
    return frozenset(p.input_content_types)


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
    # doc_processor(鏈中,預設)= documents → documents。
    "custom": SlotFactory(
        build=_build_custom_parsing,
        kind_fn=_custom_parsing_kind,
        produces_pages_fn=_custom_parsing_produces_pages,
        input_content_types_fn=_custom_parsing_input_types,
    ),
}


class _CustomChunkingParams(CustomModuleParams):
    """``chunking: method: custom`` 的參數:custom module 定位 + 分頁需求宣告。"""

    requires_pages: bool = Field(
        default=False,
        description="元件是否需要分頁輸入(供與 parsing 鏈的相容性檢查:"
        "true 時 parsing 鏈必須產生頁界)",
    )


def _build_custom_chunking(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("chunking", "custom", _CustomChunkingParams, raw)
    return instantiate_custom("chunking", p)


def _custom_chunking_requires_pages(raw: dict[str, Any]) -> bool:
    return _validate_params(
        "chunking", "custom", _CustomChunkingParams, raw
    ).requires_pages


CHUNKING_FACTORIES: dict[str, SlotFactory] = {
    "fixed_size": SlotFactory(build=_build_fixed_size),
    "structure_based": SlotFactory(build=_build_structure_based),
    "page_based": SlotFactory(build=_build_page_based, requires_pages=True),
    "no_chunking": SlotFactory(build=_build_no_chunking),
    # 契約:documents → documents(meta 逐塊繼承,doc_id 必須保留 ——
    # 下游 stamper 依 doc_id 產生穩定 chunk_id)。
    "custom": SlotFactory(
        build=_build_custom_chunking,
        requires_pages_fn=_custom_chunking_requires_pages,
    ),
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


# ---------------------------------------------------------------------------
# Ingestion pipeline 組裝
# ---------------------------------------------------------------------------


def _build_parsing_chain(cfg: MethodConfig) -> tuple[list[str], list[SlotFactory]]:
    """解析 parsing 方法鏈:鏈首必須是 converter,其餘必須是文件處理器。

    回傳的 factories 已經過 :func:`_resolved_factory`(custom 的 kind /
    produces_pages / input_content_types 從 config 參數解析成靜態欄位),
    呼叫端可直接讀宣告欄位。
    """
    methods = cfg.methods()
    factories = [
        _resolved_factory(_resolve("parsing", PARSING_FACTORIES, m), cfg.params_for(m))
        for m in methods
    ]
    if factories[0].kind != "converter":
        converters = [
            name for name, f in PARSING_FACTORIES.items() if f.kind == "converter"
        ]
        listed = ", ".join(repr(n) for n in sorted(converters))
        raise ConfigError(
            f"parsing 鏈的第一個方法必須是 converter(檔案 → Document),"
            f"但 '{methods[0]}' 不是。請把 {listed} 之一放在鏈首;"
            "custom 方法要放鏈首時,在其參數設 kind: converter"
        )
    for position, (method, factory) in enumerate(zip(methods, factories)):
        if position > 0 and factory.kind != "doc_processor":
            raise ConfigError(
                f"parsing 鏈的第 {position + 1} 個方法 '{method}' 是 converter;"
                "converter 只能放在鏈首,後續環節必須是文件處理器(如 'clean';"
                "custom 方法預設 kind: doc_processor 即可放鏈中)"
            )
    return methods, factories


def build_ingestion_pipeline(
    config: RAGConfig, *, store: Any = None
) -> tuple[Pipeline, Any]:
    """把 config 的 ingestion 槽位翻譯成 Haystack Pipeline。

    Args:
        config: 已驗證的整體配置(``config.ingestion`` 不可為 None)。
        store: 既有的 document store;None 時依 ``indexing`` 槽位建立。

    Returns:
        ``(pipeline, document_store)``;pipeline 以 ``pipeline.run({})``
        執行(來源資訊已在 config 中)。

    Raises:
        ConfigError / UnknownMethodError / IncompatiblePipelineError:
            方法不存在、參數不合法或組合不相容(建構期報錯)。
    """
    ing = config.ingestion
    if ing is None:
        raise ConfigError(
            "此配置的 ingestion 由 haystack_pipelines 提供(原生 pipeline);"
            "請改用 build_pipelines() 載入"
        )
    ctx = BuildContext(embedding_config=ing.embedding)

    indexing_method = _require_single("indexing", ing.indexing)
    indexing_factory = _resolve("indexing", INDEXING_FACTORIES, ing.indexing.methods()[0])
    ctx.indexing_method = indexing_method
    if store is None:
        store = indexing_factory.build(ing.indexing.params_for(indexing_method), ctx)
    ctx.store = store

    incremental = bool(
        ing.indexing.params_for(indexing_method).get("incremental", False)
    )
    if incremental and "incremental_update" not in indexing_factory.capabilities:
        raise IncompatiblePipelineError(
            f"indexing 方法 '{indexing_method}' 未宣告 incremental_update 能力,"
            "無法使用 incremental: true(增量 ingest 需要按 id 查找既有切片)"
        )

    import_method = _require_single("import", ing.import_)
    import_factory = _resolve("import", IMPORT_FACTORIES, import_method)
    import_params = ing.import_.params_for(import_method)
    lister = import_factory.build(import_params, ctx)  # 參數先過 pydantic 驗證
    output_content_type = (
        import_factory.output_content_type_fn(import_params)
        if import_factory.output_content_type_fn is not None
        else import_factory.output_content_type
    )

    parse_methods, parse_factories = _build_parsing_chain(ing.parsing)
    validate_ingestion_compatibility(
        import_method, output_content_type, parse_methods[0], parse_factories[0],
        PARSING_FACTORIES,
    )
    chain_produces_pages = any(f.produces_pages for f in parse_factories)

    chunking_method = _require_single("chunking", ing.chunking)
    chunking_factory = _resolved_factory(
        _resolve("chunking", CHUNKING_FACTORIES, chunking_method),
        ing.chunking.params_for(chunking_method),
    )
    validate_chunking_compatibility(
        parse_methods, chain_produces_pages, chunking_method, chunking_factory,
        PARSING_FACTORIES, CHUNKING_FACTORIES,
    )
    splitter = chunking_factory.build(ing.chunking.params_for(chunking_method), ctx)

    embedding_method = _require_single("embedding", ing.embedding)
    embedding_factory = _resolve("embedding", EMBEDDING_FACTORIES, embedding_method)
    doc_embedder, _ = embedding_factory.build(
        ing.embedding.params_for(embedding_method), ctx
    )

    pipeline = Pipeline()
    pipeline.add_component("importer", lister)

    # 檔案層增量:importer 之後就過濾,未變更的檔案連 parse(含 OCR)
    # 都不會發生。manifest 由 RagPipelines.run_ingestion 在成功後持久化。
    if incremental:
        from rag import kb_meta

        index_name = ing.indexing.params_for(indexing_method).get("index")
        config_key = kb_meta.parse_config_key(
            config.ingestion.model_dump(by_alias=True)
        )
        pipeline.add_component(
            "source_filter",
            SourceChangeFilter(
                read_previous=lambda: kb_meta.read_manifest(
                    indexing_method, store, index_name
                ),
                config_key=config_key,
            ),
        )
        pipeline.connect("importer.sources", "source_filter.sources")
        pipeline.connect("importer.meta", "source_filter.meta")
        sources_from, meta_from = "source_filter.sources", "source_filter.meta"
    else:
        sources_from, meta_from = "importer.sources", "importer.meta"

    # 鏈首 converter:單一元件,或展開為內部圖(auto 的 router + 分支)。
    head = parse_factories[0].build(ing.parsing.params_for(parse_methods[0]), ctx)
    if isinstance(head, ParsingGraph):
        def _abs(rel: str) -> str:
            return f"parser_{rel}"  # parser_router / parser_text / …

        for rel, comp in head.components.items():
            pipeline.add_component(_abs(rel), comp)
        for sender, receiver in head.connections:
            s_comp, s_sock = sender.split(".", 1)
            r_comp, r_sock = receiver.split(".", 1)
            pipeline.connect(f"{_abs(s_comp)}.{s_sock}", f"{_abs(r_comp)}.{r_sock}")
        pipeline.connect(
            sources_from, f"{_abs(head.sources_target[0])}.{head.sources_target[1]}"
        )
        pipeline.connect(
            meta_from, f"{_abs(head.meta_target[0])}.{head.meta_target[1]}"
        )
        docs_chain = [_abs(head.output)]
    else:
        pipeline.add_component("parser", head)
        pipeline.connect(sources_from, "parser.sources")
        pipeline.connect(meta_from, "parser.meta")
        docs_chain = ["parser"]

    # 鏈尾 doc_processor(clean 等):照舊命名 parser_2、parser_3…
    for position, (method, factory) in enumerate(
        zip(parse_methods[1:], parse_factories[1:]), start=2
    ):
        name = f"parser_{position}"
        pipeline.add_component(name, factory.build(ing.parsing.params_for(method), ctx))
        docs_chain.append(name)

    pipeline.add_component("stamper", ChunkMetaStamper())
    pipeline.add_component("embedder", doc_embedder)
    pipeline.add_component(
        "writer", DocumentWriter(store, policy=DuplicatePolicy.OVERWRITE)
    )
    if splitter is not None:
        pipeline.add_component("chunker", splitter)
        docs_chain.append("chunker")
    docs_chain.append("stamper")
    if incremental:
        # 蓋章後、embedding 前:內容未變的切片在這裡被擋下,省掉 embedding
        pipeline.add_component("change_filter", IncrementalChangeFilter(store))
        docs_chain.append("change_filter")
    docs_chain += ["embedder", "writer"]
    for upstream, downstream in zip(docs_chain, docs_chain[1:]):
        pipeline.connect(f"{upstream}.documents", f"{downstream}.documents")
    return pipeline, store


# ---------------------------------------------------------------------------
# Inference 槽位的 factories
# ---------------------------------------------------------------------------

# 預設 prompt 模板(Jinja2):切片帶 [chunk_id] 前綴,引用可回溯;
# glossary_notes 由 glossary 方法提供,未接線時渲染為空。
DEFAULT_PROMPT_TEMPLATE = """\
根據以下內容回答問題:
{% for doc in documents %}[{{ doc.meta.chunk_id }}] {{ doc.content }}
{% if not loop.last %}---
{% endif %}{% endfor %}
{% if glossary_notes %}術語說明:
{{ glossary_notes }}
{% endif %}問題:{{ query }}"""


class _GenCommonParams(BaseParams):
    """所有 generation 方法共用的 prompt 參數。"""

    prompt_template: str | None = Field(
        default=None,
        description="Jinja2 prompt 模板(可用 {{ query }}、{{ documents }}、"
        "{{ glossary_notes }});未設定時使用內建模板",
    )
    system_prompt: str | None = Field(default=None, description="system 角色訊息(選填)")


class _MockGenParams(_GenCommonParams):
    replies: list[str] | None = Field(
        default=None, description="腳本化回覆(依序循環);未設定時回覆可辨識的假答案"
    )


def _build_mock_generator(raw: dict[str, Any], ctx: BuildContext) -> tuple[Any, Any, Any]:
    p = _validate_params("generation", "mock", _MockGenParams, raw)
    return (MockChatGenerator(replies=p.replies), p.prompt_template, p.system_prompt)


class _OpenAIGenParams(_GenCommonParams):
    model: str = Field(default="gpt-5-mini", description="OpenAI 模型名稱")
    api_key: str | None = Field(
        default=None, description="API key;未設定時使用 OPENAI_API_KEY 環境變數"
    )
    api_base_url: str | None = Field(
        default=None, description="替代的 base URL(vLLM / Ollama / 代理閘道等)"
    )
    temperature: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout: float | None = Field(default=None, gt=0)


def _build_openai_generator(
    raw: dict[str, Any], ctx: BuildContext
) -> tuple[Any, Any, Any]:
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.utils import Secret

    p = _validate_params("generation", "openai", _OpenAIGenParams, raw)
    api_key = (
        Secret.from_token(p.api_key)
        if p.api_key
        else Secret.from_env_var("OPENAI_API_KEY")
    )
    generation_kwargs: dict[str, Any] = {}
    if p.temperature is not None:
        generation_kwargs["temperature"] = p.temperature
    if p.max_tokens is not None:
        generation_kwargs["max_tokens"] = p.max_tokens
    generator = OpenAIChatGenerator(
        api_key=api_key,
        model=p.model,
        api_base_url=p.api_base_url,
        generation_kwargs=generation_kwargs or None,
        timeout=p.timeout,
    )
    return (generator, p.prompt_template, p.system_prompt)


class _GatewayGenParams(_GenCommonParams):
    base_url: str = Field(description="API base URL,例如 https://llm.example.com/v1")
    api_key: str | None = Field(
        default=None, description="API key;建議用 ${ENV_VAR} 由環境變數注入"
    )
    model: str | None = Field(
        default=None,
        description="模型名稱;不設定時請求不帶 model 欄位(適用於內部閘道)",
    )
    temperature: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout: float = Field(default=60.0, gt=0)
    headers: dict[str, str] = Field(default_factory=dict)
    completions_path: str = Field(default="/chat/completions")


def _build_gateway_generator(
    raw: dict[str, Any], ctx: BuildContext
) -> tuple[Any, Any, Any]:
    p = _validate_params(
        "generation", "gateway_openai_compatible", _GatewayGenParams, raw
    )
    generator = GatewayChatGenerator(
        base_url=p.base_url,
        api_key=p.api_key,
        model=p.model,
        temperature=p.temperature,
        max_tokens=p.max_tokens,
        timeout=p.timeout,
        headers=p.headers,
        completions_path=p.completions_path,
    )
    return (generator, p.prompt_template, p.system_prompt)


class _CustomGenParams(CustomModuleParams, _GenCommonParams):
    """``generation: method: custom`` 的參數:custom module 定位 + prompt 欄位。

    多重繼承而非重寫欄位:``CustomModuleParams`` 提供 class_path / file /
    class / init_params 與「兩者擇一」的驗證,``_GenCommonParams`` 提供
    prompt_template / system_prompt —— prompt 由框架的 ChatPromptBuilder
    組裝(custom 元件只收 messages),因此模板設定與內建方法完全同款。
    """


def _build_custom_generator(
    raw: dict[str, Any], ctx: BuildContext
) -> tuple[Any, Any, Any]:
    p = _validate_params("generation", "custom", _CustomGenParams, raw)
    return (
        instantiate_custom("generation", p),
        p.prompt_template,
        p.system_prompt,
    )


GENERATION_FACTORIES: dict[str, SlotFactory] = {
    "mock": SlotFactory(build=_build_mock_generator),
    "openai": SlotFactory(build=_build_openai_generator),
    "gateway_openai_compatible": SlotFactory(build=_build_gateway_generator),
    # 契約:messages: list[ChatMessage] → replies: list[ChatMessage]
    # (自訂 SDK / 內部推論服務;prompt 仍由框架組)。
    "custom": SlotFactory(build=_build_custom_generator),
}


class _KeywordMatchParams(BaseParams):
    routes: dict[str, list[str]] = Field(
        description="類別 → 關鍵字清單(查詢包含關鍵字即命中)"
    )
    default_category: str = Field(
        default="general", description="沒有任何關鍵字命中時回傳的類別"
    )


def _build_keyword_match(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("routing", "keyword_match", _KeywordMatchParams, raw)
    return KeywordRouteClassifier(
        routes=p.routes, default_category=p.default_category
    )


def _build_custom_routing(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("routing", "custom", CustomModuleParams, raw)
    return instantiate_custom("routing", p)


# routing 槽位本身是選填(config 省略 = 不做),因此不提供 "none" 方法。
ROUTING_FACTORIES: dict[str, SlotFactory] = {
    "keyword_match": SlotFactory(build=_build_keyword_match),
    "custom": SlotFactory(build=_build_custom_routing),
}


class _SimpleJsonParams(BaseParams):
    include_content: bool = Field(
        default=True, description="payload 是否包含切片內文(false = 只留引用資訊)"
    )


def _build_simple_json(raw: dict[str, Any], ctx: BuildContext) -> Any:
    from rag.components.formatting import SimpleJsonFormatter

    p = _validate_params("formatter", "simple_json", _SimpleJsonParams, raw)
    return SimpleJsonFormatter(include_content=p.include_content)


def _build_custom_formatter(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("formatter", "custom", CustomModuleParams, raw)
    return instantiate_custom("formatter", p)


# formatter 槽位是選填(config 省略 = 不做),因此不提供 "none" 方法。
# 契約:documents + query → payload: Any(終端支線,型別開放見 contracts.py)。
FORMATTER_FACTORIES: dict[str, SlotFactory] = {
    "simple_json": SlotFactory(build=_build_simple_json),
    "custom": SlotFactory(build=_build_custom_formatter),
}


def _chat_generator_from_block(
    where: str, block: dict[str, Any] | None, ctx: BuildContext
) -> Any:
    """為 llm_decompose / llm rerank 建立 chat generator。

    ``block`` 形如 ``{method: <generation 槽位的任一方法>, params: {...}}``
    (mock / openai / gateway_openai_compatible / custom);未提供時沿用
    generation 槽位的設定(同一個 LLM,各自新實例)。
    """
    if block is None:
        if ctx.generation_config is None:
            raise ConfigError(
                f"{where} 需要 chat generator:請在 params.generator 指定,"
                "或提供 generation 槽位設定供沿用"
                "(generate_answer: false 時 generation 區塊仍可保留,僅作為沿用來源)"
            )
        method = ctx.generation_config.methods()[0]
        factory = _resolve("generation", GENERATION_FACTORIES, method)
        generator, _, _ = factory.build(ctx.generation_config.params_for(method), ctx)
        return generator
    if not isinstance(block, dict) or "method" not in block:
        raise ConfigError(
            f"{where} 的 params.generator 必須是 {{method, params}} 形式的物件,"
            f"實際得到:{block!r}"
        )
    method = block["method"]
    factory = _resolve("generation", GENERATION_FACTORIES, method)
    generator, _, _ = factory.build(dict(block.get("params") or {}), ctx)
    return generator


class _NormalizeParams(BaseParams):
    lowercase: bool = Field(default=True, description="是否把拉丁字母轉為小寫")


def _build_normalize(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("query_transformation", "normalize", _NormalizeParams, raw)
    return QueryNormalizer(lowercase=p.lowercase)


class _GlossaryParams(BaseParams):
    glossary: dict[str, str] | None = Field(
        default=None, description="行內術語表(術語: 定義)"
    )
    glossary_path: str | None = Field(
        default=None, description="術語表 YAML 檔路徑(格式:術語: 定義)"
    )
    expand_query: bool = Field(
        default=False, description="是否把命中的術語定義附加到查詢文字"
    )


def _build_glossary(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("query_transformation", "glossary", _GlossaryParams, raw)
    return GlossaryExpander(
        glossary=p.glossary, glossary_path=p.glossary_path, expand_query=p.expand_query
    )


class _DecomposeParams(BaseParams):
    max_subqueries: int = Field(default=4, gt=1, description="子查詢數上限")
    prompt: str = Field(
        default=DEFAULT_DECOMPOSE_PROMPT, description="拆解 prompt(含 {{ query }})"
    )
    generator: dict[str, Any] | None = Field(
        default=None,
        description="拆解用的 chat generator({method, params});"
        "未設定時沿用 generation 槽位",
    )


def _build_llm_decompose(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("query_transformation", "llm_decompose", _DecomposeParams, raw)
    chat_generator = _chat_generator_from_block(
        "query_transformation 方法 'llm_decompose'", p.generator, ctx
    )
    return LLMQueryDecomposer(
        chat_generator=chat_generator,
        prompt=p.prompt,
        max_subqueries=p.max_subqueries,
    )


class _JargonMappingParams(BaseParams):
    mapping: dict[str, str] | None = Field(
        default=None, description="行內術語對照表(術語: 直白描述)"
    )
    json_path: str | None = Field(
        default=None, description="對照表 JSON 檔路徑(扁平物件:術語: 直白描述)"
    )


def _build_jargon_mapping(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("query_transformation", "jargon_mapping", _JargonMappingParams, raw)
    return JargonMapper(mapping=p.mapping, json_path=p.json_path)


class _RewriteParams(BaseParams):
    prompt: str = Field(
        default=DEFAULT_REWRITE_PROMPT, description="改寫 prompt(含 {{ query }})"
    )
    generator: dict[str, Any] | None = Field(
        default=None,
        description="改寫用的 chat generator({method, params});"
        "未設定時沿用 generation 槽位",
    )


def _build_llm_rewrite(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("query_transformation", "llm_rewrite", _RewriteParams, raw)
    chat_generator = _chat_generator_from_block(
        "query_transformation 方法 'llm_rewrite'", p.generator, ctx
    )
    return LLMQueryRewriter(chat_generator=chat_generator, prompt=p.prompt)


class _PassthroughParams(BaseParams):
    pass


def _build_passthrough(raw: dict[str, Any], ctx: BuildContext) -> None:
    _validate_params("query_transformation", "passthrough", _PassthroughParams, raw)
    return None


def _build_custom_transform(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("query_transformation", "custom", CustomModuleParams, raw)
    return instantiate_custom("query_transformation", p)


TRANSFORM_FACTORIES: dict[str, SlotFactory] = {
    "passthrough": SlotFactory(build=_build_passthrough),
    "normalize": SlotFactory(build=_build_normalize),
    "glossary": SlotFactory(build=_build_glossary),
    "jargon_mapping": SlotFactory(build=_build_jargon_mapping),
    "llm_rewrite": SlotFactory(build=_build_llm_rewrite),
    "llm_decompose": SlotFactory(build=_build_llm_decompose),
    "custom": SlotFactory(build=_build_custom_transform),
}


class _SimilarityRankerParams(BaseParams):
    model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="cross-encoder 模型"
    )
    top_k: int = Field(default=10, gt=0)


def _build_similarity_ranker(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("reranking", "similarity", _SimilarityRankerParams, raw)
    try:
        # 2.32 起 sentence-transformers 元件移出 core,優先從整合套件 import。
        from haystack_integrations.components.rankers.sentence_transformers import (
            SentenceTransformersSimilarityRanker,
        )
    except ImportError:
        if importlib.util.find_spec("sentence_transformers") is None:
            raise MissingDependencyError(
                "sentence-transformers-haystack", "reranking 方法 'similarity'"
            ) from None
        from haystack.components.rankers import SentenceTransformersSimilarityRanker
    return SentenceTransformersSimilarityRanker(model=p.model, top_k=p.top_k)


class _LLMRerankParams(BaseParams):
    top_k: int = Field(default=5, gt=0)
    generator: dict[str, Any] | None = Field(
        default=None,
        description="重排用的 chat generator({method, params});"
        "未設定時沿用 generation 槽位",
    )


def _build_llm_ranker(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("reranking", "llm", _LLMRerankParams, raw)
    # 注意:LLMRanker 不傳 chat_generator 會在 init 時建 OpenAIChatGenerator
    # (要求 OPENAI_API_KEY);builder 一律顯式傳入。
    chat_generator = _chat_generator_from_block(
        "reranking 方法 'llm'", p.generator, ctx
    )
    return LLMRanker(chat_generator=chat_generator, top_k=p.top_k)


class _LLMFactCheckParams(BaseParams):
    prompt: str = Field(
        default=DEFAULT_FACT_CHECK_PROMPT,
        description="查核 prompt(含 {{ query }} 與 {{ documents }})",
    )
    max_docs: int | None = Field(
        default=None, gt=0, description="送交 LLM 查核的切片數上限;其餘原樣通過"
    )
    generator: dict[str, Any] | None = Field(
        default=None,
        description="查核用的 chat generator({method, params});"
        "未設定時沿用 generation 槽位",
    )


def _build_llm_fact_check(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("reranking", "llm_fact_check", _LLMFactCheckParams, raw)
    chat_generator = _chat_generator_from_block(
        "reranking 方法 'llm_fact_check'", p.generator, ctx
    )
    return LLMFactChecker(
        chat_generator=chat_generator, prompt=p.prompt, max_docs=p.max_docs
    )


class _APIRerankParams(BaseParams):
    endpoint: str = Field(description="rerank API 端點(完整 URL)")
    headers: dict[str, str] = Field(
        default_factory=dict, description="額外 HTTP 標頭(認證放這裡)"
    )
    model: str | None = Field(
        default=None, description="模型名稱;None 時請求不帶該欄位"
    )
    top_k: int = Field(default=5, gt=0)
    timeout: float = Field(default=30.0, gt=0)
    query_field: str = Field(default="question", description="請求中放查詢的欄位名")
    documents_field: str = Field(
        default="documents", description="請求中放候選文字的欄位名"
    )
    model_field: str = Field(default="model", description="請求中放模型的欄位名")
    results_field: str | None = Field(
        default="returnData",
        description="回應中結果清單的欄位(支援 a.b;回應本身是清單時設 null)",
    )
    index_field: str = Field(default="index", description="結果元素中名次索引的欄位名")
    score_field: str = Field(default="score", description="結果元素中分數的欄位名")
    index_base: int = Field(
        default=0, ge=0, le=1, description="回應 index 的起算基準(0 或 1)"
    )
    higher_is_better: bool = Field(
        default=True, description="分數越大越相關;回傳距離的 API 設 false"
    )
    raise_on_failure: bool = Field(
        default=False,
        description="API 失敗時中斷查詢;預設 false = 保留原檢索順序並記警告",
    )


def _build_api_ranker(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("reranking", "api_rerank", _APIRerankParams, raw)
    return FlexibleAPIRanker(**p.model_dump())


class _NoRerankParams(BaseParams):
    pass


def _build_no_rerank(raw: dict[str, Any], ctx: BuildContext) -> None:
    _validate_params("reranking", "none", _NoRerankParams, raw)
    return None


def _build_custom_reranker(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("reranking", "custom", CustomModuleParams, raw)
    return instantiate_custom("reranking", p)


RERANKING_FACTORIES: dict[str, SlotFactory] = {
    "none": SlotFactory(build=_build_no_rerank),
    "similarity": SlotFactory(build=_build_similarity_ranker),
    "api_rerank": SlotFactory(build=_build_api_ranker),
    "llm": SlotFactory(build=_build_llm_ranker),
    "llm_fact_check": SlotFactory(build=_build_llm_fact_check),
    "custom": SlotFactory(build=_build_custom_reranker),
}


@dataclass
class _RetrievalGraph:
    """retrieval 方法展開後的內部圖片段。"""

    components: dict[str, Any]
    connections: list[tuple[str, str]]
    query_targets: list[tuple[str, str]]
    output: str


def _retriever_classes(ctx: BuildContext) -> tuple[Any, Any]:
    """依 indexing 方法回傳 (BM25Retriever, EmbeddingRetriever) 類別。"""
    if ctx.indexing_method == "in_memory":
        from haystack.components.retrievers.in_memory import (
            InMemoryBM25Retriever,
            InMemoryEmbeddingRetriever,
        )

        return InMemoryBM25Retriever, InMemoryEmbeddingRetriever
    if ctx.indexing_method == "elasticsearch":
        try:
            from haystack_integrations.components.retrievers.elasticsearch import (
                ElasticsearchBM25Retriever,
                ElasticsearchEmbeddingRetriever,
            )
        except ImportError as exc:
            raise MissingDependencyError(
                "elasticsearch-haystack", "elasticsearch 索引的 retrieval"
            ) from exc
        return ElasticsearchBM25Retriever, ElasticsearchEmbeddingRetriever
    raise ConfigError(
        f"indexing 方法 '{ctx.indexing_method}' 沒有對應的 retriever 實作"
    )


def _query_text_embedder(ctx: BuildContext) -> Any:
    """從 ingestion.embedding 設定派生查詢端 embedder(同向量空間紀律)。"""
    if ctx.embedding_config is None:
        raise ConfigError("缺少 embedding 設定,無法建立查詢端 embedder")
    method = ctx.embedding_config.methods()[0]
    factory = _resolve("embedding", EMBEDDING_FACTORIES, method)
    _, text_embedder = factory.build(ctx.embedding_config.params_for(method), ctx)
    return text_embedder


class _RetrievalParams(BaseParams):
    top_k: int = Field(default=10, gt=0, description="取回的切片數上限")
    boost_k_factor: int = Field(
        default=1,
        ge=1,
        description="候選放大倍率:各 retriever 取回 top_k × boost_k_factor 筆,"
        "供下游 rerank 收斂到 top n",
    )

    @property
    def fetch_k(self) -> int:
        return self.top_k * self.boost_k_factor


def _build_bm25_retrieval(raw: dict[str, Any], ctx: BuildContext) -> _RetrievalGraph:
    p = _validate_params("retrieval", "bm25", _RetrievalParams, raw)
    bm25_cls, _ = _retriever_classes(ctx)
    return _RetrievalGraph(
        components={"retriever": bm25_cls(document_store=ctx.store, top_k=p.fetch_k)},
        connections=[],
        query_targets=[("retriever", "query")],
        output="retriever",
    )


def _build_embedding_retrieval(
    raw: dict[str, Any], ctx: BuildContext
) -> _RetrievalGraph:
    p = _validate_params("retrieval", "embedding", _RetrievalParams, raw)
    _, embedding_cls = _retriever_classes(ctx)
    return _RetrievalGraph(
        components={
            "query_embedder": _query_text_embedder(ctx),
            "retriever": embedding_cls(document_store=ctx.store, top_k=p.fetch_k),
        },
        connections=[("query_embedder.embedding", "retriever.query_embedding")],
        query_targets=[("query_embedder", "text")],
        output="retriever",
    )


def _build_hybrid_retrieval(raw: dict[str, Any], ctx: BuildContext) -> _RetrievalGraph:
    p = _validate_params("retrieval", "hybrid", _RetrievalParams, raw)
    bm25_cls, embedding_cls = _retriever_classes(ctx)
    # joiner 也用 fetch_k:joiner 是餵給 reranker 的輸出,不能提前裁切候選池。
    return _RetrievalGraph(
        components={
            "query_embedder": _query_text_embedder(ctx),
            "embedding_retriever": embedding_cls(
                document_store=ctx.store, top_k=p.fetch_k
            ),
            "bm25_retriever": bm25_cls(document_store=ctx.store, top_k=p.fetch_k),
            "joiner": DocumentJoiner(
                join_mode="reciprocal_rank_fusion", top_k=p.fetch_k
            ),
        },
        connections=[
            ("query_embedder.embedding", "embedding_retriever.query_embedding"),
            ("embedding_retriever.documents", "joiner.documents"),
            ("bm25_retriever.documents", "joiner.documents"),
        ],
        query_targets=[("query_embedder", "text"), ("bm25_retriever", "query")],
        output="joiner",
    )


def _build_custom_retrieval(raw: dict[str, Any], ctx: BuildContext) -> _RetrievalGraph:
    p = _validate_params("retrieval", "custom", CustomModuleParams, raw)
    comp = instantiate_custom("retrieval", p)
    return _RetrievalGraph(
        components={"retriever": comp},
        connections=[],
        query_targets=[("retriever", "query")],
        output="retriever",
    )


RETRIEVAL_FACTORIES: dict[str, SlotFactory] = {
    "bm25": SlotFactory(
        build=_build_bm25_retrieval,
        required_capabilities=frozenset({"text_search"}),
    ),
    "embedding": SlotFactory(
        build=_build_embedding_retrieval,
        required_capabilities=frozenset({"vector_search"}),
    ),
    "hybrid": SlotFactory(
        build=_build_hybrid_retrieval,
        required_capabilities=frozenset({"vector_search", "text_search"}),
    ),
    # custom 不宣告 required_capabilities:自訂 retriever 通常自帶檢索
    # 後端(如公司 ES 的 HTTP API),不依賴 ctx.store 的能力。
    "custom": SlotFactory(build=_build_custom_retrieval),
}


# ---------------------------------------------------------------------------
# Inference pipeline 組裝 / escape hatch / RagPipelines facade
# ---------------------------------------------------------------------------


def build_inference_pipeline(
    config: RAGConfig, *, store: Any = None
) -> tuple[Pipeline, dict[str, Any]]:
    """把 config 的 inference 槽位翻譯成 Haystack Pipeline。

    圖形狀::

        query → [transform 鏈] → MultiQueryRetrievalStage(內部:retriever
        → ranker 鏈) → SubqueryFusion → ChatPromptBuilder → generator

    ``generate_answer: false`` 時省略 ChatPromptBuilder 與 generator,
    圖止於 SubqueryFusion(檢索-only)。

    Returns:
        ``(pipeline, meta)``;``meta["query_entry"]`` 是查詢文字的入口
        socket(RagPipelines.query() 用),``meta["store"]`` 是實際使用的
        document store(呼叫時沒給 ``store`` 就是這裡依 indexing 槽位建的)。

    Raises:
        ConfigError / UnknownMethodError / IncompatiblePipelineError
    """
    inf = config.inference
    if inf is None:
        raise ConfigError(
            "此配置的 inference 由 haystack_pipelines 提供(原生 pipeline);"
            "請改用 build_pipelines() 載入"
        )
    ing = config.ingestion
    if ing is None:
        raise ConfigError(
            "inference 槽位需要 ingestion 槽位提供索引與 embedding 設定;"
            "ingestion 使用原生 pipeline 時,inference 也請以 "
            "haystack_pipelines 提供"
        )
    ctx = BuildContext(
        embedding_config=ing.embedding, generation_config=inf.generation
    )
    ctx.indexing_method = _require_single("indexing", ing.indexing)
    indexing_factory = _resolve("indexing", INDEXING_FACTORIES, ctx.indexing_method)
    if store is None:
        store = indexing_factory.build(
            ing.indexing.params_for(ctx.indexing_method), ctx
        )
    ctx.store = store

    # --- 內部單查詢 pipeline:retrieval + reranking 鏈 ---
    retrieval_method = _require_single("retrieval", inf.retrieval)
    retrieval_factory = _resolve("retrieval", RETRIEVAL_FACTORIES, retrieval_method)
    validate_inference_compatibility(
        retrieval_method,
        retrieval_factory,
        ctx.indexing_method,
        indexing_factory,
        RETRIEVAL_FACTORIES,
    )
    graph = retrieval_factory.build(
        inf.retrieval.params_for(retrieval_method), ctx
    )

    inner = Pipeline()
    for name, comp in graph.components.items():
        inner.add_component(name, comp)
    for sender, receiver in graph.connections:
        inner.connect(sender, receiver)
    query_targets = list(graph.query_targets)
    docs_output = graph.output

    ranker_index = 0
    for method in inf.reranking.methods():
        factory = _resolve("reranking", RERANKING_FACTORIES, method)
        ranker = factory.build(inf.reranking.params_for(method), ctx)
        if ranker is None:
            continue
        ranker_index += 1
        name = "ranker" if ranker_index == 1 else f"ranker_{ranker_index}"
        inner.add_component(name, ranker)
        inner.connect(f"{docs_output}.documents", f"{name}.documents")
        query_targets.append((name, "query"))
        docs_output = name

    stage = MultiQueryRetrievalStage(
        inner=inner, query_targets=query_targets, output_component=docs_output
    )

    # --- 外部 pipeline:transform 鏈 → multi_query → fusion → prompt → LLM ---
    outer = Pipeline()
    transform_names: list[str] = []
    glossary_name: str | None = None
    for position, method in enumerate(inf.query_transformation.methods()):
        factory = _resolve("query_transformation", TRANSFORM_FACTORIES, method)
        transform = factory.build(inf.query_transformation.params_for(method), ctx)
        if transform is None:  # passthrough
            continue
        name = "transform" if not transform_names else f"transform_{position + 1}"
        outer.add_component(name, transform)
        if transform_names:
            outer.connect(f"{transform_names[-1]}.queries", f"{name}.queries")
        transform_names.append(name)
        if isinstance(transform, GlossaryExpander):
            if glossary_name is not None:
                raise ConfigError(
                    "query_transformation 鏈中最多只能有一個 'glossary' 方法"
                    "(prompt 只有一個 glossary_notes 輸入)"
                )
            glossary_name = name

    outer.add_component("multi_query", stage)
    if transform_names:
        outer.connect(f"{transform_names[-1]}.queries", "multi_query.queries")
    query_entry = (
        (transform_names[0], "queries") if transform_names else ("multi_query", "queries")
    )

    fusion_cfg = inf.fusion or FusionConfig()
    if fusion_cfg.method == "custom":
        # 掛上即一律執行:單一查詢的 N=1 也進元件,原樣通過與否由元件
        # 自己決定(建議額外輸出 applied: bool 供 trace 區分)。
        fusion_params = _validate_params(
            "fusion", "custom", CustomModuleParams, fusion_cfg.params
        )
        fusion_component: Any = instantiate_custom("fusion", fusion_params)
    else:
        fusion_component = SubqueryFusion(
            group_by=fusion_cfg.group_by,
            strategy=fusion_cfg.strategy,
            top_k=fusion_cfg.top_k,
            always_fuse=inf.fusion is not None,
        )
    outer.add_component("fusion", fusion_component)
    outer.connect("multi_query.results", "fusion.results")

    # routing:獨立支線 —— 吃原始查詢(query() 直接餵,如同 prompt_builder
    # 的 query),圖上沒有下游;判斷結果由 query() 收進回傳值與 trace。
    routing_enabled = False
    if inf.routing is not None:
        routing_method = _require_single("routing", inf.routing)
        routing_factory = _resolve("routing", ROUTING_FACTORIES, routing_method)
        outer.add_component(
            "routing",
            routing_factory.build(inf.routing.params_for(routing_method), ctx),
        )
        routing_enabled = True

    # formatter:終端支線 —— 從 fusion.documents 岔出(與 prompt → generator
    # 並聯,fan-out 合法),把最終結果組成對外格式;query 由 query() 執行期
    # 餵入。payload 由 query() 收進回傳值的 output 鍵,圖上沒有下游。
    formatter_enabled = False
    if inf.formatter is not None:
        formatter_method = _require_single("formatter", inf.formatter)
        formatter_factory = _resolve(
            "formatter", FORMATTER_FACTORIES, formatter_method
        )
        outer.add_component(
            "formatter",
            formatter_factory.build(inf.formatter.params_for(formatter_method), ctx),
        )
        outer.connect("fusion.documents", "formatter.documents")
        formatter_enabled = True

    if inf.generate_answer:
        generation_method = _require_single("generation", inf.generation)
        generation_factory = _resolve(
            "generation", GENERATION_FACTORIES, generation_method
        )
        generator, prompt_template, system_prompt = generation_factory.build(
            inf.generation.params_for(generation_method), ctx
        )
        template_messages: list[ChatMessage] = []
        if system_prompt:
            template_messages.append(ChatMessage.from_system(system_prompt))
        template_messages.append(
            ChatMessage.from_user(prompt_template or DEFAULT_PROMPT_TEMPLATE)
        )
        outer.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                template=template_messages, required_variables=["query", "documents"]
            ),
        )
        outer.add_component("generator", generator)
        outer.connect("fusion.documents", "prompt_builder.documents")
        if glossary_name is not None:
            outer.connect(f"{glossary_name}.notes", "prompt_builder.glossary_notes")
        outer.connect("prompt_builder.prompt", "generator.messages")
    return outer, {
        "query_entry": query_entry,
        "generate_answer": inf.generate_answer,
        "transform_names": transform_names,
        "routing_enabled": routing_enabled,
        "formatter_enabled": formatter_enabled,
        "store": store,
    }


def _load_native_pipeline(path: str) -> Pipeline:
    """載入 escape hatch 指定的原生 Haystack pipeline YAML。"""
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"找不到 haystack_pipelines 指定的檔案:{file}")
    try:
        return Pipeline.loads(file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(
            f"無法載入原生 Haystack pipeline '{file}':{type(exc).__name__}: {exc}"
        ) from exc


@dataclass
class RagPipelines:
    """一組建好的 pipelines(共用同一個 document store)。

    ``stage`` 記錄這組是為哪個階段建的(見 :func:`build_pipelines`):
    ``"ingestion"`` 時 ``inference`` 為 ``None``、``"inference"`` 時
    ``ingestion`` 為 ``None``,對應的 :meth:`run_ingestion` / :meth:`query`
    會報錯並指明要怎麼重建 —— 半條 pipeline 被誤用時,錯在建構意圖上,
    不該讓它變成執行期一個難解的 KeyError。
    """

    config: RAGConfig
    ingestion: Pipeline | None
    inference: Pipeline | None
    store: Any
    query_entry: tuple[str, str] | None = None
    generate_answer: bool = True
    transform_names: list[str] = dc_field(default_factory=list)
    routing_enabled: bool = False
    formatter_enabled: bool = False
    stage: str = "all"

    def run_ingestion(self) -> dict[str, Any]:
        """執行 ingestion(來源資訊已在 config 中)。

        Raises:
            ConfigError: 這組 pipelines 以 ``stage="inference"`` 建立,
                沒有 ingestion pipeline 可執行。

        Returns:
            dict:各元件的輸出(``importer`` / ``parser`` / ``chunker`` /
            ``stamper`` / ``embedder`` / ``writer`` …),外加 ``trace``
            —— 依執行順序列出每一步的元件名稱、類別與產出筆數,
            供 CLI 或稽核時逐步檢視。
        """
        if self.ingestion is None:
            raise ConfigError(
                f"這組 pipelines 以 stage='{self.stage}' 建立,沒有 ingestion "
                "pipeline 可執行。需要建索引請改用 "
                "build_pipelines(config, stage='ingestion')(或 stage='all')"
            )
        step_names = step_order(self.ingestion)
        result = self.ingestion.run({}, include_outputs_from=set(step_names))

        # 檔案層增量:成功跑完才持久化 manifest(失敗不會誤記「已處理」)
        source_output = result.get("source_filter") or {}
        manifest = source_output.get("manifest")
        if manifest is not None and self.config is not None:
            from rag import kb_meta

            method, index_name = kb_meta.indexing_info(self.config)
            kb_meta.write_manifest(method, self.store, manifest, index_name)

        # 顯式回報沒有產出任何切片的來源檔案(掃描檔沒 OCR、空檔、
        # 解析失敗被跳過…)—— 靜默消失的檔案是最難查的問題。
        listed = [
            entry.get("doc_id")
            for entry in (result.get("importer") or {}).get("meta") or []
        ]
        skipped_files = set(source_output.get("skipped_files") or [])
        stamped = {
            doc.meta.get("doc_id")
            for doc in (result.get("stamper") or {}).get("documents") or []
        }
        empty_sources = [
            doc_id
            for doc_id in listed
            if doc_id and doc_id not in stamped and doc_id not in skipped_files
        ]
        result["empty_sources"] = empty_sources
        if empty_sources:
            logger.warning(
                "以下來源檔案沒有產出任何切片(掃描檔需要 OCR?空檔?"
                "解析失敗?詳見上方紀錄):%s", ", ".join(empty_sources),
            )

        trace = []
        for name in step_names:
            output = result.get(name) or {}
            # 防禦:router 有來源落進未接線的 socket = 檔案被靜默丟棄。
            # 理論上不會發生(FileLister 已按副檔名預過濾),發生就要出聲。
            for key in ("unclassified", "failed"):
                leftover = output.get(key)
                if leftover:
                    logger.warning(
                        "元件 '%s' 有 %d 個來源進入 '%s'(未被任何分支處理):%s",
                        name, len(leftover), key, [str(item)[:80] for item in leftover[:5]],
                    )
            payload = next(
                (output[key] for key in ("documents", "sources") if key in output), None
            )
            trace.append(
                {
                    "component": name,
                    "type": type(self.ingestion.get_component(name)).__name__,
                    "count": len(payload) if payload is not None else None,
                    "output": output,
                }
            )
        result["trace"] = trace
        return result

    def query(self, text: str) -> dict[str, Any]:
        """執行單次查詢,回傳答案與可稽核的中間結果。

        Returns:
            dict:``answer``(回答文字)、``documents``(融合後切片)、
            ``subquery_results``(各子查詢重排後結果)、``prompt``
            (實際送出的 prompt,可稽核)、``reply_meta``(模型 /
            finish_reason / usage 等)、``routing``(查詢分類結果;
            未設定 routing 槽位時為 ``None``)、``output``(formatter
            組出的對外格式 payload;未設定 formatter 槽位時為 ``None``)、
            ``trace``(逐步紀錄,見下)。
            檢索-only 模式(``generate_answer: false``)key 不變,
            但 ``answer`` / ``prompt`` / ``reply_meta`` 為 ``None``。

            ``trace`` 的形狀::

                {
                  "query": 原始查詢,
                  "transforms": [{component, type, queries}],   # 每段改寫後的查詢
                  "routing": {category, ...} | None,            # 查詢分類(獨立支線)
                  "subqueries": [{query, steps: [{component, type, documents}]}],
                  "fusion": {documents, applied},               # 融合 / 去重後
                                                # applied 三態:True 融合過 /
                                                # False 內建穿透 / None custom 未回報
                  "formatter": {type, payload} | None,          # 對外格式(終端支線)
                  "generation": {prompt, answer, meta} | None,
                }

        Raises:
            ConfigError: 這組 pipelines 以 ``stage="ingestion"`` 建立
                (沒有 inference pipeline),或 inference 由原生 Haystack
                YAML 載入(query() 便利介面只支援槽位式配置,請直接呼叫
                ``.inference.run(...)``)。
        """
        if self.inference is None:
            raise ConfigError(
                f"這組 pipelines 以 stage='{self.stage}' 建立,沒有 inference "
                "pipeline 可查詢。需要查詢請改用 "
                "build_pipelines(config, stage='inference')(或 stage='all')"
            )
        if self.query_entry is None:
            raise ConfigError(
                "inference pipeline 由原生 Haystack YAML 載入,"
                "query() 便利介面不適用;請直接呼叫 .inference.run(...)"
            )
        entry_component, entry_socket = self.query_entry
        data: dict[str, Any] = {entry_component: {entry_socket: [text]}}
        # 全元件輸出:transform 鏈的每段改寫、內部檢索 / 重排的每步結果
        # 都要留下紀錄,查詢過程才是可稽核的而非黑箱。
        include = set(self.inference.graph.nodes)
        if self.generate_answer:
            data["prompt_builder"] = {"query": text}
        if self.routing_enabled:
            # routing 吃原始查詢(不經 transform 鏈),與 prompt_builder 同機制
            data["routing"] = {"query": text}
        if self.formatter_enabled:
            # formatter 的 documents 接自 fusion;query 與 prompt_builder 同機制
            data["formatter"] = {"query": text}
        result = self.inference.run(data, include_outputs_from=include)
        route = (
            (result.get("routing") or {}).get("route")
            if self.routing_enabled
            else None
        )
        output = (
            (result.get("formatter") or {}).get("payload")
            if self.formatter_enabled
            else None
        )
        answer = prompt_text = reply_meta = None
        if self.generate_answer:
            reply = result["generator"]["replies"][0]
            prompt_messages = result.get("prompt_builder", {}).get("prompt", [])
            prompt_text = "\n\n".join(
                message.text or "" for message in prompt_messages
            )
            answer = reply.text
            reply_meta = dict(reply.meta or {})
        transforms = [
            {
                "component": name,
                "type": type(self.inference.get_component(name)).__name__,
                "queries": list((result.get(name) or {}).get("queries", [])),
            }
            for name in self.transform_names
        ]
        return {
            "answer": answer,
            "documents": result["fusion"]["documents"],
            "subquery_results": result["multi_query"]["results"],
            "prompt": prompt_text,
            "reply_meta": reply_meta,
            "routing": route,
            "output": output,
            "trace": {
                "query": text,
                "transforms": transforms,
                "routing": route,
                "subqueries": result["multi_query"].get("traces", []),
                "fusion": {
                    "documents": result["fusion"]["documents"],
                    # 三態:True = 融合過;False = 內建穿透(單一查詢且
                    # config 未設 fusion);None = custom 元件未回報
                    # (custom fusion 一律執行,applied 是建議輸出)
                    "applied": result["fusion"].get("applied"),
                },
                "formatter": (
                    {
                        "type": type(
                            self.inference.get_component("formatter")
                        ).__name__,
                        "payload": output,
                    }
                    if self.formatter_enabled
                    else None
                ),
                "generation": (
                    {"prompt": prompt_text, "answer": answer, "meta": reply_meta}
                    if self.generate_answer
                    else None
                ),
            },
        }


def build_pipelines(
    config: RAGConfig, *, store: Any = None, stage: str = "all"
) -> RagPipelines:
    """把整份配置翻譯成可執行的 pipelines(含 escape hatch 處理)。

    槽位式的階段經 builder 組裝;``haystack_pipelines`` 指定的階段直接
    載入原生 pipeline YAML(跳過相容性檢查,專家模式)。

    Args:
        config: 已驗證的整體配置。
        store: 既有的 document store;None 時依 ``indexing`` 槽位建立。
        stage: 要組哪些階段。**只決定建不建那條 pipeline,不改變任何一條
            的組法**,因此半條與整條的行為完全一致:

            - ``"all"``(預設):兩條都組。
            - ``"ingestion"``:只建索引。省掉 inference 側元件(reranker
              的 cross-encoder、查詢端 embedder…)的載入;:meth:`RagPipelines.query`
              會報錯。
            - ``"inference"``:只查詢,索引沿用既有內容。省掉 ingestion
              側元件(document embedder…)與「來源資料夾必須存在」的前提;
              :meth:`RagPipelines.run_ingestion` 會報錯。適用於索引已建好
              (elasticsearch)、或 retrieval 走 custom 外部檢索時。

    Raises:
        ConfigError: ``stage`` 不是上述三者之一。

    注意:escape hatch 的 ingestion + 槽位式 inference 只在索引為外部
    服務(如 elasticsearch)時有意義 —— in_memory store 無法跨 pipeline
    共享原生 pipeline 寫入的內容。
    """
    if stage not in PIPELINE_STAGES:
        raise ConfigError(
            f"未知的 stage '{stage}';可用的值:"
            f"{', '.join(repr(s) for s in PIPELINE_STAGES)}"
        )
    native = config.haystack_pipelines

    ingestion_pipeline: Pipeline | None = None
    if stage != "inference":
        if native is not None and native.ingestion is not None:
            ingestion_pipeline = _load_native_pipeline(native.ingestion)
        else:
            ingestion_pipeline, store = build_ingestion_pipeline(config, store=store)

    inference_pipeline: Pipeline | None = None
    query_entry: tuple[str, str] | None = None
    generate_answer = True
    transform_names: list[str] = []
    routing_enabled = False
    formatter_enabled = False
    if stage != "ingestion":
        if native is not None and native.inference is not None:
            inference_pipeline = _load_native_pipeline(native.inference)
        else:
            inference_pipeline, meta = build_inference_pipeline(config, store=store)
            # stage="inference" 時 store 是 inference 側依 indexing 槽位建的,
            # 要收回來(kb_meta 的指紋 / manifest 都掛在 store 上)。
            store = meta["store"]
            query_entry = meta["query_entry"]
            generate_answer = meta["generate_answer"]
            transform_names = meta["transform_names"]
            routing_enabled = meta["routing_enabled"]
            formatter_enabled = meta["formatter_enabled"]
    return RagPipelines(
        config=config,
        ingestion=ingestion_pipeline,
        inference=inference_pipeline,
        store=store,
        query_entry=query_entry,
        generate_answer=generate_answer,
        transform_names=transform_names,
        routing_enabled=routing_enabled,
        formatter_enabled=formatter_enabled,
        stage=stage,
    )

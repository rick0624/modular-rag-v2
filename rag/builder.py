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

from dataclasses import dataclass, field as dc_field
from typing import Any, Callable

from haystack import Pipeline
from haystack.components.preprocessors import (
    DocumentCleaner,
    DocumentSplitter,
    RecursiveDocumentSplitter,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag.compatibility import (
    validate_chunking_compatibility,
    validate_ingestion_compatibility,
)
from rag.components.api_embedders import (
    FlexibleAPIDocumentEmbedder,
    FlexibleAPITextEmbedder,
)
from rag.components.file_lister import FileLister
from rag.components.meta_stamper import ChunkMetaStamper
from rag.components.mock_embedders import MockDocumentEmbedder, MockTextEmbedder
from rag.config import MethodConfig, RAGConfig
from rag.errors import ConfigError, MissingDependencyError, UnknownMethodError

# 輸入輸出同型別、支援方法鏈(method 清單)的槽位。
CHAINABLE_SLOTS = frozenset({"parsing", "query_transformation", "reranking"})


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
    output_content_type: str | None = None  # import 槽位宣告
    input_content_types: frozenset[str] = frozenset()  # parsing 槽位宣告
    produces_pages: bool = False  # parsing 槽位宣告
    requires_pages: bool = False  # chunking 槽位宣告
    capabilities: frozenset[str] = frozenset()  # indexing 槽位宣告
    required_capabilities: frozenset[str] = frozenset()  # retrieval 槽位宣告


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


class _PdfParams(BaseParams):
    pass


def _build_pdf(raw: dict[str, Any], ctx: BuildContext) -> Any:
    from haystack.components.converters.pypdf import PyPDFToDocument

    _validate_params("parsing", "pdf", _PdfParams, raw)
    return PyPDFToDocument()


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


class _InMemoryParams(BaseParams):
    pass


def _build_in_memory_store(raw: dict[str, Any], ctx: BuildContext) -> Any:
    _validate_params("indexing", "in_memory", _InMemoryParams, raw)
    return InMemoryDocumentStore(embedding_similarity_function="cosine")


class _ElasticsearchParams(BaseParams):
    hosts: str | list[str] = Field(description="ES 端點,如 http://localhost:9200")
    index: str = Field(default="modular-rag", description="索引名稱")
    api_key: str | None = Field(default=None, description="API key(選填)")


def _build_elasticsearch_store(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = _validate_params("indexing", "elasticsearch", _ElasticsearchParams, raw)
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
    }
    if p.api_key is not None:
        kwargs["api_key"] = p.api_key
    return ElasticsearchDocumentStore(**kwargs)


IMPORT_FACTORIES: dict[str, SlotFactory] = {
    "local_file": SlotFactory(
        build=_make_file_lister_builder("local_file", [".txt", ".md"]),
        output_content_type="text",
    ),
    "pdf_file": SlotFactory(
        build=_make_file_lister_builder("pdf_file", [".pdf"]),
        output_content_type="pdf",
    ),
}

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
    "clean": SlotFactory(build=_build_clean, kind="doc_processor"),
}

CHUNKING_FACTORIES: dict[str, SlotFactory] = {
    "fixed_size": SlotFactory(build=_build_fixed_size),
    "structure_based": SlotFactory(build=_build_structure_based),
    "page_based": SlotFactory(build=_build_page_based, requires_pages=True),
    "no_chunking": SlotFactory(build=_build_no_chunking),
}

EMBEDDING_FACTORIES: dict[str, SlotFactory] = {
    "mock": SlotFactory(build=_build_mock_embedding),
    "api_embedding": SlotFactory(build=_build_api_embedding),
    "sentence_transformers": SlotFactory(build=_build_sentence_transformers),
}

INDEXING_FACTORIES: dict[str, SlotFactory] = {
    "in_memory": SlotFactory(
        build=_build_in_memory_store,
        capabilities=frozenset({"vector_search", "text_search", "metadata_filter"}),
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
    """解析 parsing 方法鏈:鏈首必須是 converter,其餘必須是文件處理器。"""
    methods = cfg.methods()
    factories = [_resolve("parsing", PARSING_FACTORIES, m) for m in methods]
    if factories[0].kind != "converter":
        converters = [
            name for name, f in PARSING_FACTORIES.items() if f.kind == "converter"
        ]
        listed = ", ".join(repr(n) for n in sorted(converters))
        raise ConfigError(
            f"parsing 鏈的第一個方法必須是 converter(檔案 → Document),"
            f"但 '{methods[0]}' 不是。請把 {listed} 之一放在鏈首"
        )
    for position, (method, factory) in enumerate(zip(methods, factories)):
        if position > 0 and factory.kind != "doc_processor":
            raise ConfigError(
                f"parsing 鏈的第 {position + 1} 個方法 '{method}' 是 converter;"
                "converter 只能放在鏈首,後續環節必須是文件處理器(如 'clean')"
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

    import_method = _require_single("import", ing.import_)
    import_factory = _resolve("import", IMPORT_FACTORIES, import_method)
    lister = import_factory.build(ing.import_.params_for(import_method), ctx)

    parse_methods, parse_factories = _build_parsing_chain(ing.parsing)
    validate_ingestion_compatibility(
        import_method, import_factory, parse_methods[0], parse_factories[0],
        PARSING_FACTORIES,
    )
    chain_produces_pages = any(f.produces_pages for f in parse_factories)

    chunking_method = _require_single("chunking", ing.chunking)
    chunking_factory = _resolve("chunking", CHUNKING_FACTORIES, chunking_method)
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
    parser_names: list[str] = []
    for position, (method, factory) in enumerate(zip(parse_methods, parse_factories)):
        name = "parser" if position == 0 else f"parser_{position + 1}"
        parser_names.append(name)
        pipeline.add_component(
            name, factory.build(ing.parsing.params_for(method), ctx)
        )
    pipeline.add_component("stamper", ChunkMetaStamper())
    pipeline.add_component("embedder", doc_embedder)
    pipeline.add_component(
        "writer", DocumentWriter(store, policy=DuplicatePolicy.OVERWRITE)
    )

    pipeline.connect("importer.sources", f"{parser_names[0]}.sources")
    pipeline.connect("importer.meta", f"{parser_names[0]}.meta")
    docs_chain = list(parser_names)
    if splitter is not None:
        pipeline.add_component("chunker", splitter)
        docs_chain.append("chunker")
    docs_chain += ["stamper", "embedder", "writer"]
    for upstream, downstream in zip(docs_chain, docs_chain[1:]):
        pipeline.connect(f"{upstream}.documents", f"{downstream}.documents")
    return pipeline, store

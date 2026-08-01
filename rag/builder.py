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

import importlib.util
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable

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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag.compatibility import (
    validate_chunking_compatibility,
    validate_inference_compatibility,
    validate_ingestion_compatibility,
)
from rag.components.api_embedders import (
    FlexibleAPIDocumentEmbedder,
    FlexibleAPITextEmbedder,
)
from rag.components.file_lister import FileLister
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
from rag.config import FusionConfig, MethodConfig, RAGConfig
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
    # BM25 斷詞:預設的 \w+ 會把整段中文吞成一個 token,中文查詢幾乎
    # 比不到;改為「CJK 逐字 + 拉丁/數字詞」(ES 的 standard analyzer
    # 對 CJK 也是逐字,行為一致)。
    return InMemoryDocumentStore(
        embedding_similarity_function="cosine",
        bm25_tokenization_regex=(
            r"[㐀-䶿一-鿿豈-﫿]|[A-Za-z0-9_]+"
        ),
    )


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


GENERATION_FACTORIES: dict[str, SlotFactory] = {
    "mock": SlotFactory(build=_build_mock_generator),
    "openai": SlotFactory(build=_build_openai_generator),
    "gateway_openai_compatible": SlotFactory(build=_build_gateway_generator),
}


def _chat_generator_from_block(
    where: str, block: dict[str, Any] | None, ctx: BuildContext
) -> Any:
    """為 llm_decompose / llm rerank 建立 chat generator。

    ``block`` 形如 ``{method: mock|openai|gateway_openai_compatible, params: {...}}``;
    未提供時沿用 generation 槽位的設定(同一個 LLM,各自新實例)。
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


TRANSFORM_FACTORIES: dict[str, SlotFactory] = {
    "passthrough": SlotFactory(build=_build_passthrough),
    "normalize": SlotFactory(build=_build_normalize),
    "glossary": SlotFactory(build=_build_glossary),
    "jargon_mapping": SlotFactory(build=_build_jargon_mapping),
    "llm_rewrite": SlotFactory(build=_build_llm_rewrite),
    "llm_decompose": SlotFactory(build=_build_llm_decompose),
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


class _NoRerankParams(BaseParams):
    pass


def _build_no_rerank(raw: dict[str, Any], ctx: BuildContext) -> None:
    _validate_params("reranking", "none", _NoRerankParams, raw)
    return None


RERANKING_FACTORIES: dict[str, SlotFactory] = {
    "none": SlotFactory(build=_build_no_rerank),
    "similarity": SlotFactory(build=_build_similarity_ranker),
    "llm": SlotFactory(build=_build_llm_ranker),
    "llm_fact_check": SlotFactory(build=_build_llm_fact_check),
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
        socket(RagPipelines.query() 用)。

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
    outer.add_component(
        "fusion",
        SubqueryFusion(
            group_by=fusion_cfg.group_by,
            strategy=fusion_cfg.strategy,
            top_k=fusion_cfg.top_k,
            always_fuse=inf.fusion is not None,
        ),
    )
    outer.connect("multi_query.results", "fusion.results")

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
    return outer, {"query_entry": query_entry, "generate_answer": inf.generate_answer}


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
    """一組建好的 pipelines(共用同一個 document store)。"""

    config: RAGConfig
    ingestion: Pipeline
    inference: Pipeline
    store: Any
    query_entry: tuple[str, str] | None = None
    generate_answer: bool = True

    def run_ingestion(self) -> dict[str, Any]:
        """執行 ingestion(來源資訊已在 config 中)。"""
        return self.ingestion.run({})

    def query(self, text: str) -> dict[str, Any]:
        """執行單次查詢,回傳答案與可稽核的中間結果。

        Returns:
            dict:``answer``(回答文字)、``documents``(融合後切片)、
            ``subquery_results``(各子查詢重排後結果)、``prompt``
            (實際送出的 prompt,可稽核)、``reply_meta``(模型 /
            finish_reason / usage 等)。檢索-only 模式
            (``generate_answer: false``)key 不變,但 ``answer`` /
            ``prompt`` / ``reply_meta`` 為 ``None``。

        Raises:
            ConfigError: inference 由原生 Haystack YAML 載入
                (query() 便利介面只支援槽位式配置,請直接呼叫
                ``.inference.run(...)``)。
        """
        if self.query_entry is None:
            raise ConfigError(
                "inference pipeline 由原生 Haystack YAML 載入,"
                "query() 便利介面不適用;請直接呼叫 .inference.run(...)"
            )
        entry_component, entry_socket = self.query_entry
        data: dict[str, Any] = {entry_component: {entry_socket: [text]}}
        include = {"fusion", "multi_query"}
        if self.generate_answer:
            data["prompt_builder"] = {"query": text}
            include.add("prompt_builder")
        result = self.inference.run(data, include_outputs_from=include)
        answer = prompt_text = reply_meta = None
        if self.generate_answer:
            reply = result["generator"]["replies"][0]
            prompt_messages = result.get("prompt_builder", {}).get("prompt", [])
            prompt_text = "\n\n".join(
                message.text or "" for message in prompt_messages
            )
            answer = reply.text
            reply_meta = dict(reply.meta or {})
        return {
            "answer": answer,
            "documents": result["fusion"]["documents"],
            "subquery_results": result["multi_query"]["results"],
            "prompt": prompt_text,
            "reply_meta": reply_meta,
        }


def build_pipelines(config: RAGConfig, *, store: Any = None) -> RagPipelines:
    """把整份配置翻譯成可執行的 pipelines(含 escape hatch 處理)。

    槽位式的階段經 builder 組裝;``haystack_pipelines`` 指定的階段直接
    載入原生 pipeline YAML(跳過相容性檢查,專家模式)。

    注意:escape hatch 的 ingestion + 槽位式 inference 只在索引為外部
    服務(如 elasticsearch)時有意義 —— in_memory store 無法跨 pipeline
    共享原生 pipeline 寫入的內容。
    """
    native = config.haystack_pipelines
    if native is not None and native.ingestion is not None:
        ingestion_pipeline = _load_native_pipeline(native.ingestion)
    else:
        ingestion_pipeline, store = build_ingestion_pipeline(config, store=store)

    query_entry: tuple[str, str] | None = None
    generate_answer = True
    if native is not None and native.inference is not None:
        inference_pipeline = _load_native_pipeline(native.inference)
    else:
        inference_pipeline, meta = build_inference_pipeline(config, store=store)
        query_entry = meta["query_entry"]
        generate_answer = meta["generate_answer"]
    return RagPipelines(
        config=config,
        ingestion=ingestion_pipeline,
        inference=inference_pipeline,
        store=store,
        query_entry=query_entry,
        generate_answer=generate_answer,
    )

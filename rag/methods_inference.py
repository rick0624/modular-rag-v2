"""Inference 方法型錄:query_transformation / retrieval / reranking /
generation / routing / formatter。

每個方法 = 一個參數 schema(pydantic)+ 一個 build 函式 + 對映表一行;
新增方法時照這個模式加一組即可,組裝邏輯(rag/builder.py)不需改動。
llm_* 方法未指定 ``params.generator`` 時,沿用 generation 槽位的設定
(:func:`_chat_generator_from_block`,各自新實例)。
"""

from __future__ import annotations

import importlib.util
from typing import Any

from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import LLMRanker
from pydantic import Field

from rag.components.api_clients import FlexibleAPIRanker
from rag.components.fact_check import DEFAULT_FACT_CHECK_PROMPT, LLMFactChecker
from rag.components.gateway_generator import GatewayChatGenerator, MockChatGenerator
from rag.components.llm_rerankers import DEFAULT_INSERTRANK_PROMPT, InsertRankLLMRanker
from rag.components.query_transforms import (
    DEFAULT_DECOMPOSE_PROMPT,
    DEFAULT_MULTI_HYDE_PROMPT,
    DEFAULT_PREQRAG_CLASSIFY_PROMPT,
    DEFAULT_PREQRAG_REWRITE_PROMPT,
    DEFAULT_REWRITE_PROMPT,
    GlossaryExpander,
    JargonMapper,
    LLMMultiHyDEExpander,
    LLMQueryDecomposer,
    LLMQueryRewriter,
    PreQRAGDispatcher,
    QueryNormalizer,
)
from rag.components.side_branches import KeywordRouteClassifier, SimpleJsonFormatter
from rag.custom import CustomModuleParams, instantiate_custom
from rag.errors import ConfigError, MissingDependencyError
from rag.methods_ingestion import EMBEDDING_FACTORIES
from rag.slots import (
    BaseParams,
    BuildContext,
    SlotFactory,
    SlotGraph,
    resolve,
    validate_params,
)

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
    p = validate_params("generation", "mock", _MockGenParams, raw)
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

    p = validate_params("generation", "openai", _OpenAIGenParams, raw)
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
    p = validate_params(
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
    p = validate_params("generation", "custom", _CustomGenParams, raw)
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
    p = validate_params("routing", "keyword_match", _KeywordMatchParams, raw)
    return KeywordRouteClassifier(
        routes=p.routes, default_category=p.default_category
    )


def _build_custom_routing(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("routing", "custom", CustomModuleParams, raw)
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
    p = validate_params("formatter", "simple_json", _SimpleJsonParams, raw)
    return SimpleJsonFormatter(include_content=p.include_content)


def _build_custom_formatter(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("formatter", "custom", CustomModuleParams, raw)
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
        factory = resolve("generation", GENERATION_FACTORIES, method)
        generator, _, _ = factory.build(ctx.generation_config.params_for(method), ctx)
        return generator
    if not isinstance(block, dict) or "method" not in block:
        raise ConfigError(
            f"{where} 的 params.generator 必須是 {{method, params}} 形式的物件,"
            f"實際得到:{block!r}"
        )
    method = block["method"]
    factory = resolve("generation", GENERATION_FACTORIES, method)
    generator, _, _ = factory.build(dict(block.get("params") or {}), ctx)
    return generator


class _NormalizeParams(BaseParams):
    lowercase: bool = Field(default=True, description="是否把拉丁字母轉為小寫")


def _build_normalize(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("query_transformation", "normalize", _NormalizeParams, raw)
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
    p = validate_params("query_transformation", "glossary", _GlossaryParams, raw)
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
    p = validate_params("query_transformation", "llm_decompose", _DecomposeParams, raw)
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
    p = validate_params("query_transformation", "jargon_mapping", _JargonMappingParams, raw)
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
    p = validate_params("query_transformation", "llm_rewrite", _RewriteParams, raw)
    chat_generator = _chat_generator_from_block(
        "query_transformation 方法 'llm_rewrite'", p.generator, ctx
    )
    return LLMQueryRewriter(chat_generator=chat_generator, prompt=p.prompt)


class _MultiHyDEParams(BaseParams):
    num_documents: int = Field(default=3, ge=1, description="假設文件篇數")
    keep_original: bool = Field(
        default=True,
        description="是否保留原查詢(原查詢 + 假設文件各自檢索後融合);"
        "false = 只用假設文件檢索",
    )
    prompt: str = Field(
        default=DEFAULT_MULTI_HYDE_PROMPT,
        description="假設文件 prompt(含 {{ query }} 與 {num_documents})",
    )
    generator: dict[str, Any] | None = Field(
        default=None,
        description="生成用的 chat generator({method, params});"
        "未設定時沿用 generation 槽位",
    )


def _build_llm_multi_hyde(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("query_transformation", "llm_multi_hyde", _MultiHyDEParams, raw)
    chat_generator = _chat_generator_from_block(
        "query_transformation 方法 'llm_multi_hyde'", p.generator, ctx
    )
    return LLMMultiHyDEExpander(
        chat_generator=chat_generator,
        prompt=p.prompt,
        num_documents=p.num_documents,
        keep_original=p.keep_original,
    )


class _PreQRAGParams(BaseParams):
    num_rewrites: int = Field(default=2, gt=0, description="single 分支的改寫條數")
    max_subqueries: int = Field(default=4, gt=1, description="multi 分支的子查詢數上限")
    include_original: bool = Field(
        default=True, description="是否保留原查詢(與改寫/子查詢各自檢索後融合)"
    )
    classify_prompt: str = Field(
        default=DEFAULT_PREQRAG_CLASSIFY_PROMPT,
        description="分類 prompt(含 {{ query }})",
    )
    rewrite_prompt: str = Field(
        default=DEFAULT_PREQRAG_REWRITE_PROMPT,
        description="改寫 prompt(含 {{ query }} 與 {num_rewrites})",
    )
    decompose_prompt: str = Field(
        default=DEFAULT_DECOMPOSE_PROMPT,
        description="拆解 prompt(含 {{ query }} 與 {max_subqueries})",
    )
    generator: dict[str, Any] | None = Field(
        default=None,
        description="分類/改寫/拆解共用的 chat generator({method, params});"
        "未設定時沿用 generation 槽位",
    )


def _build_preqrag(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("query_transformation", "preqrag", _PreQRAGParams, raw)
    chat_generator = _chat_generator_from_block(
        "query_transformation 方法 'preqrag'", p.generator, ctx
    )
    return PreQRAGDispatcher(
        chat_generator=chat_generator,
        classify_prompt=p.classify_prompt,
        rewrite_prompt=p.rewrite_prompt,
        decompose_prompt=p.decompose_prompt,
        num_rewrites=p.num_rewrites,
        max_subqueries=p.max_subqueries,
        include_original=p.include_original,
    )


class _PassthroughParams(BaseParams):
    pass


def _build_passthrough(raw: dict[str, Any], ctx: BuildContext) -> None:
    validate_params("query_transformation", "passthrough", _PassthroughParams, raw)
    return None


def _build_custom_transform(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("query_transformation", "custom", CustomModuleParams, raw)
    return instantiate_custom("query_transformation", p)


TRANSFORM_FACTORIES: dict[str, SlotFactory] = {
    "passthrough": SlotFactory(build=_build_passthrough),
    "normalize": SlotFactory(build=_build_normalize),
    "glossary": SlotFactory(build=_build_glossary),
    "jargon_mapping": SlotFactory(build=_build_jargon_mapping),
    "llm_rewrite": SlotFactory(build=_build_llm_rewrite),
    "llm_decompose": SlotFactory(build=_build_llm_decompose),
    "llm_multi_hyde": SlotFactory(build=_build_llm_multi_hyde),
    "preqrag": SlotFactory(build=_build_preqrag),
    "custom": SlotFactory(build=_build_custom_transform),
}


class _SimilarityRankerParams(BaseParams):
    model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="cross-encoder 模型"
    )
    top_k: int = Field(default=10, gt=0)


def _build_similarity_ranker(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("reranking", "similarity", _SimilarityRankerParams, raw)
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
    p = validate_params("reranking", "llm", _LLMRerankParams, raw)
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
    p = validate_params("reranking", "llm_fact_check", _LLMFactCheckParams, raw)
    chat_generator = _chat_generator_from_block(
        "reranking 方法 'llm_fact_check'", p.generator, ctx
    )
    return LLMFactChecker(
        chat_generator=chat_generator, prompt=p.prompt, max_docs=p.max_docs
    )


class _InsertRankParams(BaseParams):
    top_k: int = Field(default=5, gt=0)
    score_label: str = Field(
        default="檢索分數",
        description="prompt 中分數的名稱;依上游檢索器據實描述"
        "(如 BM25 分數、RRF 融合分數)",
    )
    prompt: str = Field(
        default=DEFAULT_INSERTRANK_PROMPT,
        description="重排 prompt(含 {{ query }}、{{ documents }} 與 {score_label})",
    )
    generator: dict[str, Any] | None = Field(
        default=None,
        description="重排用的 chat generator({method, params});"
        "未設定時沿用 generation 槽位",
    )


def _build_insertrank(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("reranking", "insertrank", _InsertRankParams, raw)
    chat_generator = _chat_generator_from_block(
        "reranking 方法 'insertrank'", p.generator, ctx
    )
    return InsertRankLLMRanker(
        chat_generator=chat_generator,
        prompt=p.prompt,
        top_k=p.top_k,
        score_label=p.score_label,
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
    p = validate_params("reranking", "api_rerank", _APIRerankParams, raw)
    return FlexibleAPIRanker(**p.model_dump())


class _NoRerankParams(BaseParams):
    pass


def _build_no_rerank(raw: dict[str, Any], ctx: BuildContext) -> None:
    validate_params("reranking", "none", _NoRerankParams, raw)
    return None


def _build_custom_reranker(raw: dict[str, Any], ctx: BuildContext) -> Any:
    p = validate_params("reranking", "custom", CustomModuleParams, raw)
    return instantiate_custom("reranking", p)


RERANKING_FACTORIES: dict[str, SlotFactory] = {
    "none": SlotFactory(build=_build_no_rerank),
    "similarity": SlotFactory(build=_build_similarity_ranker),
    "api_rerank": SlotFactory(build=_build_api_ranker),
    "llm": SlotFactory(build=_build_llm_ranker),
    "insertrank": SlotFactory(build=_build_insertrank),
    "llm_fact_check": SlotFactory(build=_build_llm_fact_check),
    "custom": SlotFactory(build=_build_custom_reranker),
}


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
    factory = resolve("embedding", EMBEDDING_FACTORIES, method)
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


def _build_bm25_retrieval(raw: dict[str, Any], ctx: BuildContext) -> SlotGraph:
    p = validate_params("retrieval", "bm25", _RetrievalParams, raw)
    bm25_cls, _ = _retriever_classes(ctx)
    return SlotGraph(
        components={"retriever": bm25_cls(document_store=ctx.store, top_k=p.fetch_k)},
        connections=[],
        inputs={"query": [("retriever", "query")]},
        output="retriever",
    )


def _build_embedding_retrieval(raw: dict[str, Any], ctx: BuildContext) -> SlotGraph:
    p = validate_params("retrieval", "embedding", _RetrievalParams, raw)
    _, embedding_cls = _retriever_classes(ctx)
    return SlotGraph(
        components={
            "query_embedder": _query_text_embedder(ctx),
            "retriever": embedding_cls(document_store=ctx.store, top_k=p.fetch_k),
        },
        connections=[("query_embedder.embedding", "retriever.query_embedding")],
        inputs={"query": [("query_embedder", "text")]},
        output="retriever",
    )


def _build_hybrid_retrieval(raw: dict[str, Any], ctx: BuildContext) -> SlotGraph:
    p = validate_params("retrieval", "hybrid", _RetrievalParams, raw)
    bm25_cls, embedding_cls = _retriever_classes(ctx)
    # joiner 也用 fetch_k:joiner 是餵給 reranker 的輸出,不能提前裁切候選池。
    return SlotGraph(
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
        inputs={"query": [("query_embedder", "text"), ("bm25_retriever", "query")]},
        output="joiner",
    )


def _build_custom_retrieval(raw: dict[str, Any], ctx: BuildContext) -> SlotGraph:
    p = validate_params("retrieval", "custom", CustomModuleParams, raw)
    comp = instantiate_custom("retrieval", p)
    return SlotGraph(
        components={"retriever": comp},
        connections=[],
        inputs={"query": [("retriever", "query")]},
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



"""薄 builder:把槽位式 config 翻譯成 Haystack Pipeline,並提供執行入口。

這是主流程檔,從頭讀到尾就是資料怎麼流動:

- :func:`build_ingestion_pipeline`:import → parsing 鏈 → chunking →
  蓋章 → embedding → 寫入(含增量兩層的接線)。
- :func:`build_inference_pipeline`:transform 鏈 → 多子查詢檢索(內部
  retrieval + reranking 鏈)→ fusion → prompt → generation(含 routing /
  formatter 支線)。
- :class:`RagPipelines`:執行 facade —— ``run_ingestion()`` / ``query()``,
  組回傳值與 trace。

「方法名稱 → 元件」的對映表在 :mod:`rag.methods_ingestion` 與
:mod:`rag.methods_inference`(本檔 re-export,方便 import 與測試替換);
新增一個方法 = 在型錄檔寫一個 factory + 對映表加一行,本檔不需改動。
語意層的相容性檢查(content_type / requires_pages / 索引能力)在
建構期執行,不合法的組合直接報錯並列出可用替代。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy

from rag.compatibility import (
    validate_chunking_compatibility,
    validate_inference_compatibility,
    validate_ingestion_compatibility,
)
from rag.components.fusion import SubqueryFusion
from rag.components.ingestion_steps import (
    ChunkMetaStamper,
    IncrementalChangeFilter,
    MetaFieldMapper,
    SourceChangeFilter,
)
from rag.components.multi_query import MultiQueryRetrievalStage
from rag.components.query_transforms import GlossaryExpander
from rag.config import FusionConfig, MethodConfig, RAGConfig
from rag.custom import CustomModuleParams, instantiate_custom
from rag.errors import ConfigError, IncompatiblePipelineError
from rag.methods_ingestion import (
    CHUNKING_FACTORIES,
    EMBEDDING_FACTORIES,
    IMPORT_FACTORIES,
    INDEXING_FACTORIES,
    PARSING_FACTORIES,
    chunking_provides_fields,
    chunking_requires_pages,
    parsing_declaration,
)
from rag.methods_inference import (
    DEFAULT_PROMPT_TEMPLATE,
    FORMATTER_FACTORIES,
    GENERATION_FACTORIES,
    RERANKING_FACTORIES,
    RETRIEVAL_FACTORIES,
    ROUTING_FACTORIES,
    TRANSFORM_FACTORIES,
)
from rag.slots import (
    BaseParams,
    BuildContext,
    SlotFactory,
    SlotGraph,
    require_single,
    resolve,
    validate_params,
)
from rag.trace import step_order

logger = logging.getLogger(__name__)

# build_pipelines(stage=…) 可組的階段。順序即 CLI 的顯示順序。
PIPELINE_STAGES: tuple[str, ...] = ("all", "ingestion", "inference")


# ---------------------------------------------------------------------------
# Ingestion pipeline 組裝
# ---------------------------------------------------------------------------


def _mount_graph(
    pipeline: Pipeline, graph: SlotGraph, prefix: str = ""
) -> Callable[[str], str]:
    """把 :class:`SlotGraph` 的元件與內部接線掛進 pipeline。

    元件名稱加 ``prefix``(如 parsing 的 ``parser_``,得到 parser_router /
    parser_join…);回傳「相對名 → 絕對名」的轉換函式,呼叫端用它接
    外部輸入與輸出。
    """

    def absolute(rel: str) -> str:
        return f"{prefix}{rel}"

    for rel, comp in graph.components.items():
        pipeline.add_component(absolute(rel), comp)
    for sender, receiver in graph.connections:
        s_comp, s_sock = sender.split(".", 1)
        r_comp, r_sock = receiver.split(".", 1)
        pipeline.connect(f"{absolute(s_comp)}.{s_sock}", f"{absolute(r_comp)}.{r_sock}")
    return absolute


def _build_parsing_chain(
    cfg: MethodConfig,
) -> tuple[list[str], list[SlotFactory], list[tuple[str, bool, frozenset[str]]]]:
    """解析 parsing 方法鏈:鏈首必須是 converter,其餘必須是文件處理器。

    回傳 ``(methods, factories, declarations)``;declarations 是每個環節的
    ``(kind, produces_pages, input_content_types)``(custom 的宣告由
    :func:`parsing_declaration` 從 config 參數讀出),呼叫端據此做
    相容性檢查,不需要認識 custom。
    """
    methods = cfg.methods()
    factories = [resolve("parsing", PARSING_FACTORIES, m) for m in methods]
    declarations = [
        parsing_declaration(m, f, cfg.params_for(m))
        for m, f in zip(methods, factories)
    ]
    if declarations[0][0] != "converter":
        converters = [
            name for name, f in PARSING_FACTORIES.items() if f.kind == "converter"
        ]
        listed = ", ".join(repr(n) for n in sorted(converters))
        raise ConfigError(
            f"parsing 鏈的第一個方法必須是 converter(檔案 → Document),"
            f"但 '{methods[0]}' 不是。請把 {listed} 之一放在鏈首;"
            "custom 方法要放鏈首時,在其參數設 kind: converter"
        )
    for position, (method, (kind, _, _)) in enumerate(zip(methods, declarations)):
        if position > 0 and kind != "doc_processor":
            raise ConfigError(
                f"parsing 鏈的第 {position + 1} 個方法 '{method}' 是 converter;"
                "converter 只能放在鏈首,後續環節必須是文件處理器(如 'clean';"
                "custom 方法預設 kind: doc_processor 即可放鏈中)"
            )
    return methods, factories, declarations


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
    ctx = BuildContext(embedding_config=ing.embedding)

    indexing_method = require_single("indexing", ing.indexing)
    indexing_factory = resolve("indexing", INDEXING_FACTORIES, ing.indexing.methods()[0])
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

    import_method = require_single("import", ing.import_)
    import_factory = resolve("import", IMPORT_FACTORIES, import_method)
    import_params = ing.import_.params_for(import_method)
    lister = import_factory.build(import_params, ctx)  # 參數先過 pydantic 驗證
    output_content_type = (
        import_factory.output_content_type_fn(import_params)
        if import_factory.output_content_type_fn is not None
        else None
    )

    parse_methods, parse_factories, parse_declarations = _build_parsing_chain(
        ing.parsing
    )
    validate_ingestion_compatibility(
        import_method, output_content_type, parse_methods[0],
        parse_declarations[0][2], PARSING_FACTORIES,
    )
    chain_produces_pages = any(pages for _, pages, _ in parse_declarations)

    chunking_method = require_single("chunking", ing.chunking)
    chunking_factory = resolve("chunking", CHUNKING_FACTORIES, chunking_method)
    chunking_params = ing.chunking.params_for(chunking_method)
    validate_chunking_compatibility(
        parse_methods, chain_produces_pages, chunking_method,
        chunking_requires_pages(chunking_method, chunking_factory, chunking_params),
        PARSING_FACTORIES, CHUNKING_FACTORIES,
    )
    splitter = chunking_factory.build(chunking_params, ctx)

    embedding_method = require_single("embedding", ing.embedding)
    embedding_factory = resolve("embedding", EMBEDDING_FACTORIES, embedding_method)
    doc_embedder, _ = embedding_factory.build(
        ing.embedding.params_for(embedding_method), ctx
    )

    # --- 欄位引用檢查:embedding 的輸入欄位與 indexing 的 fields ---
    # (params 已在各 factory.build 過 pydantic 驗證,這裡讀 raw dict 安全)
    source_field = ing.embedding.params_for(embedding_method).get(
        "source_field", "content"
    )
    extra_vectors: dict[str, str] = (
        ing.embedding.params_for(embedding_method).get("extra_vectors") or {}
    )
    es_fields = ing.indexing.params_for(indexing_method).get("fields")
    declared = chunking_provides_fields(
        chunking_method, chunking_factory, chunking_params
    )
    if declared is not None:
        # source / importer 由 FileLister 固定提供,不需宣告。
        available = {"content", "doc_id", "chunk_id", "seq", "page",
                     "source", "importer"} | set(declared)
        listed = ", ".join(repr(name) for name in sorted(available))
        if source_field not in available:
            raise ConfigError(
                f"embedding 的 source_field '{source_field}' 不在 chunking 宣告的"
                f"欄位中。可用欄位:{listed}"
            )
        for vec_name, vec_source in extra_vectors.items():
            if vec_source not in available:
                raise ConfigError(
                    f"embedding 的 extra_vectors 來源欄位 '{vec_source}'"
                    f"(向量欄位 '{vec_name}')不在 chunking 宣告的欄位中。"
                    f"可用欄位:{listed}"
                )
        vec_collisions = sorted(set(extra_vectors) & available)
        if vec_collisions:
            raise ConfigError(
                f"embedding 的 extra_vectors 向量欄位名 {vec_collisions} 與"
                "既有欄位重名(向量會覆蓋 meta 中的原值),請改用其他名字"
            )
        # fields 另可引用 extra_vectors 的向量欄位(改名用);沒列的
        # 向量欄位不受白名單影響,一律原名保留。
        indexable = available | set(extra_vectors)
        listed_indexable = ", ".join(repr(name) for name in sorted(indexable))
        for es_name, meta_name in (es_fields or {}).items():
            if meta_name not in indexable:
                raise ConfigError(
                    f"indexing 的 fields 引用了未宣告的 meta 欄位 '{meta_name}'"
                    f"(ES 欄位 '{es_name}')。可用欄位:{listed_indexable}"
                )
    embed_sources = [source_field, *extra_vectors.values()]
    if incremental and es_fields is not None:
        unlisted = sorted(
            {src for src in embed_sources
             if src != "content" and src not in es_fields.values()}
        )
        if unlisted:
            names = ", ".join(repr(src) for src in unlisted)
            raise ConfigError(
                f"incremental: true 且 embedding 以 {names} 為輸入時,"
                "indexing 的 fields 白名單必須包含該欄位(寫進索引才能在下次"
                "ingest 比對它是否變更)"
            )

    # 被 embed 的欄位在 store 端的名字:可能被 fields 改名後才寫入。
    def _stored_name(field: str) -> str:
        if es_fields is None or field == "content":
            return field
        return next(
            (es for es, meta in es_fields.items() if meta == field), field
        )

    stored_field = _stored_name(source_field)
    # extra_vectors 各來源欄位的 (切片端, store 端) 名字對,供增量比對;
    # content 已是主比對項,不重複列。
    extra_change_fields = [
        (src, _stored_name(src))
        for src in extra_vectors.values()
        if src != "content"
    ]

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
    if isinstance(head, SlotGraph):
        absolute = _mount_graph(pipeline, head, prefix="parser_")  # parser_router / …
        (sources_comp, sources_sock), = head.inputs["sources"]
        (meta_comp, meta_sock), = head.inputs["meta"]
        pipeline.connect(sources_from, f"{absolute(sources_comp)}.{sources_sock}")
        pipeline.connect(meta_from, f"{absolute(meta_comp)}.{meta_sock}")
        docs_chain = [absolute(head.output)]
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
        pipeline.add_component(
            "change_filter",
            IncrementalChangeFilter(
                store,
                source_field=source_field,
                stored_field=stored_field,
                extra_fields=extra_change_fields,
            ),
        )
        docs_chain.append("change_filter")
    docs_chain.append("embedder")
    if es_fields is not None:
        # 寫入前的欄位白名單 + 改名(僅自訂欄位;框架欄位固定保留)。
        # extra_vectors 的向量欄位:列入 fields 的走映射(可改名),
        # 沒列的自動保留原名 —— 不因白名單而被丟棄。
        preserve = tuple(
            name for name in extra_vectors if name not in es_fields.values()
        )
        pipeline.add_component(
            "field_mapper",
            MetaFieldMapper(es_fields, preserve=preserve),
        )
        docs_chain.append("field_mapper")
    docs_chain.append("writer")
    for upstream, downstream in zip(docs_chain, docs_chain[1:]):
        pipeline.connect(f"{upstream}.documents", f"{downstream}.documents")
    return pipeline, store


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
    ing = config.ingestion
    ctx = BuildContext(
        embedding_config=ing.embedding, generation_config=inf.generation
    )
    ctx.indexing_method = require_single("indexing", ing.indexing)
    indexing_factory = resolve("indexing", INDEXING_FACTORIES, ctx.indexing_method)
    if store is None:
        store = indexing_factory.build(
            ing.indexing.params_for(ctx.indexing_method), ctx
        )
    ctx.store = store

    # --- 內部單查詢 pipeline:retrieval + reranking 鏈 ---
    retrieval_method = require_single("retrieval", inf.retrieval)
    retrieval_factory = resolve("retrieval", RETRIEVAL_FACTORIES, retrieval_method)
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
    _mount_graph(inner, graph)  # 不加 prefix:retriever / query_embedder / joiner
    query_targets = list(graph.inputs["query"])
    docs_output = graph.output

    ranker_index = 0
    for method in inf.reranking.methods():
        factory = resolve("reranking", RERANKING_FACTORIES, method)
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
        factory = resolve("query_transformation", TRANSFORM_FACTORIES, method)
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
        fusion_params = validate_params(
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
        routing_method = require_single("routing", inf.routing)
        routing_factory = resolve("routing", ROUTING_FACTORIES, routing_method)
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
        formatter_method = require_single("formatter", inf.formatter)
        formatter_factory = resolve(
            "formatter", FORMATTER_FACTORIES, formatter_method
        )
        outer.add_component(
            "formatter",
            formatter_factory.build(inf.formatter.params_for(formatter_method), ctx),
        )
        outer.connect("fusion.documents", "formatter.documents")
        formatter_enabled = True

    if inf.generate_answer:
        generation_method = require_single("generation", inf.generation)
        generation_factory = resolve(
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
        logger.info("ingestion 開始:%s", " → ".join(step_names))
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
            component_type = type(self.ingestion.get_component(name)).__name__
            # 每步都留一行紀錄(成功也記):事後查「哪一步之後筆數不對」
            # 不必開 --trace 重跑。
            logger.info(
                "ingestion 步驟 %s(%s):%s",
                name, component_type,
                f"產出 {len(payload)} 筆" if payload is not None
                else f"完成(輸出鍵:{sorted(output.keys()) or '無'})",
            )
            trace.append(
                {
                    "component": name,
                    "type": component_type,
                    "count": len(payload) if payload is not None else None,
                    "output": output,
                }
            )
        result["trace"] = trace
        written = (result.get("writer") or {}).get("documents_written") or 0
        logger.info(
            "ingestion 完成:寫入 %d 筆%s", written,
            f";{len(empty_sources)} 個來源沒有產出切片" if empty_sources else "",
        )
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
                (沒有 inference pipeline)。
        """
        if self.inference is None:
            raise ConfigError(
                f"這組 pipelines 以 stage='{self.stage}' 建立,沒有 inference "
                "pipeline 可查詢。需要查詢請改用 "
                "build_pipelines(config, stage='inference')(或 stage='all')"
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
        logger.info("query 開始:%r", text)
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

        # 每階段都留一行紀錄(成功也記):fail-soft 降級的細節由各元件的
        # WARNING 補充,這裡是「每步實際產出什麼」的骨架。
        for step in transforms:
            logger.info(
                "query 步驟 %s(%s):%d 條查詢",
                step["component"], step["type"], len(step["queries"]),
            )
        subquery_traces = result["multi_query"].get("traces", [])
        for sub in subquery_traces:
            steps = sub.get("steps") or []
            logger.info(
                "query 子查詢 %r:%s", sub.get("query"),
                f"經 {len(steps)} 步,最終 {len(steps[-1]['documents'])} 筆"
                if steps else "無步驟紀錄",
            )
        logger.info(
            "query 融合:%d 筆(applied=%s)",
            len(result["fusion"]["documents"]), result["fusion"].get("applied"),
        )
        if self.routing_enabled:
            logger.info("query routing:%s", route)
        if self.formatter_enabled:
            logger.info("query formatter:payload 型別 %s", type(output).__name__)
        if self.generate_answer:
            logger.info(
                "query 生成:回答 %d 字(finish_reason=%s)",
                len(answer or ""), (reply_meta or {}).get("finish_reason"),
            )
        else:
            logger.info("query 檢索-only:未生成答案(generate_answer: false)")

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
    """把整份配置翻譯成可執行的 pipelines。

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
    """
    if stage not in PIPELINE_STAGES:
        raise ConfigError(
            f"未知的 stage '{stage}';可用的值:"
            f"{', '.join(repr(s) for s in PIPELINE_STAGES)}"
        )
    ingestion_pipeline: Pipeline | None = None
    if stage != "inference":
        ingestion_pipeline, store = build_ingestion_pipeline(config, store=store)

    inference_pipeline: Pipeline | None = None
    query_entry: tuple[str, str] | None = None
    generate_answer = True
    transform_names: list[str] = []
    routing_enabled = False
    formatter_enabled = False
    if stage != "ingestion":
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

"""modular-rag-v2:配置驅動的 RAG 框架(Haystack 2.x 薄層)。

公開 API:

- :func:`load_config` / :func:`parse_config`:載入並驗證槽位式 YAML 配置。
- :func:`build_pipelines`:把配置翻譯成 Haystack pipelines
  (回傳 :class:`RagPipelines`,含 ``run_ingestion()`` 與 ``query()``)。
- :func:`build_ingestion_pipeline` / :func:`build_inference_pipeline`:
  單獨組裝某一階段(進階用法)。
"""

from rag.builder import (
    RagPipelines,
    build_ingestion_pipeline,
    build_inference_pipeline,
    build_pipelines,
)
from rag.config import RAGConfig, load_config, parse_config

__all__ = [
    "RAGConfig",
    "RagPipelines",
    "build_ingestion_pipeline",
    "build_inference_pipeline",
    "build_pipelines",
    "load_config",
    "parse_config",
]

"""modular-rag-v2:配置驅動的 RAG 框架(Haystack 2.x 薄層)。

公開 API:

- :func:`load_config` / :func:`parse_config`:載入並驗證槽位式 YAML 配置。
- ``build_pipelines``(於 builder 完成後提供):把配置翻譯成 Haystack pipelines。
"""

from rag.config import RAGConfig, load_config, parse_config

__all__ = ["RAGConfig", "load_config", "parse_config"]

"""modular-rag-v2:配置驅動的 RAG 框架(Haystack 2.x 薄層)。

公開 API:

- :func:`load_config` / :func:`parse_config`:載入並驗證槽位式 YAML 配置。
- :func:`build_pipelines`:把配置翻譯成 Haystack pipelines
  (回傳 :class:`RagPipelines`,含 ``run_ingestion()`` 與 ``query()``)。
- :func:`build_ingestion_pipeline` / :func:`build_inference_pipeline`:
  單獨組裝某一階段(進階用法)。
"""

import os

# Haystack 匯入時就會依這個變數決定是否啟用匿名遙測(連往 eu.i.posthog.com),
# 封閉網路下會拋連線錯誤並拖慢啟動,故預設關閉。
# setdefault 保留使用者既有設定;想開啟就在啟動前設 HAYSTACK_TELEMETRY_ENABLED=True。
# 必須在下方 import haystack(經 rag.builder)之前執行,不可移到 import 之後;
# 同理也無法改用 .env,因為 .env 要到 load_config() 才載入,那時已經太晚。
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")

from rag.builder import (  # noqa: E402
    RagPipelines,
    build_ingestion_pipeline,
    build_inference_pipeline,
    build_pipelines,
)
from rag.config import RAGConfig, load_config, parse_config  # noqa: E402

__all__ = [
    "RAGConfig",
    "RagPipelines",
    "build_ingestion_pipeline",
    "build_inference_pipeline",
    "build_pipelines",
    "load_config",
    "parse_config",
]

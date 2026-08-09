"""Evaluation:測試集載入與檢索指標(hit rate / MRR)。

資料集格式(JSONL,每行一筆)::

    {"query": "...", "relevant_doc_ids": ["doc_id", ...], "reference_answer": "..."}

``reference_answer`` 選填。評估器吃 :meth:`RagPipelines.query` 的完整
輸出(不只 doc_id)。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.builder import RagPipelines
from rag.config import RAGConfig
from rag.errors import ComponentError, ConfigError, UnknownMethodError
from rag.slots import BaseParams, validate_params


class EvalCase(BaseModel):
    """測試集的一筆:查詢 + 相關文件 doc_id(+ 選填參考答案)。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    relevant_doc_ids: list[str]
    reference_answer: str | None = None


def load_cases(path: str | Path) -> list[EvalCase]:
    """載入 JSONL 測試集;格式錯誤時指出檔案與 1-based 行號。

    Raises:
        ComponentError: 檔案不存在、任一行不是合法 JSON / 欄位不符,
            或資料集為空。
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise ComponentError(f"找不到評估資料集:{dataset_path}")
    cases: list[EvalCase] = []
    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            cases.append(EvalCase.model_validate(data))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ComponentError(
                f"評估資料集 '{dataset_path}' 第 {line_number} 行格式不正確:{exc}"
            ) from exc
    if not cases:
        raise ComponentError(f"評估資料集 '{dataset_path}' 沒有任何測試案例")
    return cases


def _retrieved_doc_ids(result: dict[str, Any]) -> list[str]:
    """依名次序取出去重後的 doc_id(同文件多切片只算一次)。"""
    seen: list[str] = []
    for doc in result["documents"]:
        doc_id = doc.meta.get("doc_id") or doc.id
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


class RetrievalMetricsEvaluator:
    """基本檢索指標:hit rate 與 MRR(以 doc_id 計)。"""

    def __init__(
        self,
        dataset_path: str | None = None,
        cases: list[EvalCase] | None = None,
    ) -> None:
        if dataset_path is None and not cases:
            raise ConfigError(
                "evaluation 方法 'basic_retrieval_metrics' 需要 dataset_path"
                "(JSONL 檔)或行內 cases 其中之一"
            )
        self.dataset_path = dataset_path
        self.cases = list(cases or [])

    def load_cases(self) -> list[EvalCase]:
        if self.cases:
            return self.cases
        return load_cases(self.dataset_path)  # type: ignore[arg-type]

    def evaluate(
        self, cases: list[EvalCase], results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """逐題比對檢索結果與標註的相關文件。

        - ``hit``:前 k 名(融合後結果)中任一相關文件出現即為命中。
        - ``reciprocal_rank``:第一個相關文件名次的倒數(未出現為 0)。
        """
        per_query: list[dict[str, Any]] = []
        for case, result in zip(cases, results):
            retrieved = _retrieved_doc_ids(result)
            relevant = set(case.relevant_doc_ids)
            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    reciprocal_rank = 1.0 / rank
                    break
            per_query.append(
                {
                    "query": case.query,
                    "relevant_doc_ids": case.relevant_doc_ids,
                    "retrieved_doc_ids": retrieved,
                    "hit": reciprocal_rank > 0,
                    "reciprocal_rank": reciprocal_rank,
                }
            )
        num_cases = len(per_query)
        hit_rate = sum(1 for row in per_query if row["hit"]) / num_cases
        mrr = sum(row["reciprocal_rank"] for row in per_query) / num_cases
        return {
            "metrics": {"hit_rate": hit_rate, "mrr": mrr},
            "per_query": per_query,
            "num_cases": num_cases,
        }


class _RetrievalMetricsParams(BaseParams):
    dataset_path: str | None = Field(default=None, description="JSONL 測試集路徑")
    cases: list[dict[str, Any]] | None = Field(
        default=None, description="行內測試案例(取代 dataset_path)"
    )


def build_evaluator(config: RAGConfig) -> RetrievalMetricsEvaluator:
    """依 ``config.evaluation`` 建立評估器(目前僅 basic_retrieval_metrics)。

    Raises:
        ConfigError: 配置沒有 evaluation 區塊。
        UnknownMethodError: 指定的方法不存在(列出可用方法)。
    """
    if config.evaluation is None:
        raise ConfigError("配置沒有 evaluation 區塊,無法建立評估器")
    method = config.evaluation.methods()[0]
    if len(config.evaluation.methods()) > 1:
        raise ConfigError(
            "模組 'evaluation' 不支援方法鏈;請指定單一方法"
        )
    if method != "basic_retrieval_metrics":
        raise UnknownMethodError("evaluation", method, ["basic_retrieval_metrics"])
    p = validate_params(
        "evaluation", "basic_retrieval_metrics", _RetrievalMetricsParams,
        config.evaluation.params_for(method),
    )
    inline = [EvalCase.model_validate(case) for case in (p.cases or [])]
    return RetrievalMetricsEvaluator(dataset_path=p.dataset_path, cases=inline)


def run_evaluation(pipelines: RagPipelines, config: RAGConfig) -> dict[str, Any]:
    """載入測試集、逐題執行查詢、回傳指標。"""
    evaluator = build_evaluator(config)
    cases = evaluator.load_cases()
    results = [pipelines.query(case.query) for case in cases]
    return evaluator.evaluate(cases, results)

"""demo 與測試共用的內建小語料(離線可跑,不存在時才寫入)。"""

from __future__ import annotations

import json
from pathlib import Path

SAMPLE_DOCS: dict[str, str] = {
    "faiss.txt": (
        "FAISS 是 Facebook AI 開源的向量相似度檢索函式庫,支援多種索引結構,"
        "包括 Flat、IVF 與 HNSW,並可使用 GPU 加速。\n\n"
        "在 RAG 系統中,FAISS 常用來為文件切片的向量建立索引,"
        "查詢時以近似最近鄰搜尋取回語意相近的切片。"
    ),
    "elasticsearch.txt": (
        "Elasticsearch 是分散式搜尋引擎,原生支援 BM25 關鍵字檢索、"
        "kNN 向量檢索與 metadata 過濾,索引常駐於伺服器端。\n\n"
        "混合檢索(hybrid)同時執行 BM25 與向量兩路查詢,"
        "再以 RRF(Reciprocal Rank Fusion)融合名次,兼顧精確匹配與語意相似。"
    ),
    "rag.txt": (
        "RAG(Retrieval-Augmented Generation,檢索增強生成)先從知識庫檢索"
        "相關內容,再把內容連同問題一起交給大型語言模型生成回答。\n\n"
        "常見的品質手段包括查詢拆解(query decomposition)、"
        "重排(reranking)與按文件或頁面聚合檢索結果。"
    ),
}

SAMPLE_QA: list[dict] = [
    {"query": "FAISS 支援哪些索引結構?", "relevant_doc_ids": ["faiss.txt"]},
    {"query": "混合檢索是怎麼融合兩路結果的?", "relevant_doc_ids": ["elasticsearch.txt"]},
    {"query": "什麼是檢索增強生成?", "relevant_doc_ids": ["rag.txt"]},
]


def ensure_sample_data(base_dir: str | Path = "./data") -> tuple[Path, Path]:
    """確保範例語料與評估集存在(已存在的檔案不覆寫)。

    Returns:
        ``(raw_dir, qa_path)``:語料資料夾與 JSONL 評估集路徑。
    """
    base = Path(base_dir)
    raw_dir = base / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, content in SAMPLE_DOCS.items():
        path = raw_dir / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    qa_path = base / "eval" / "qa.jsonl"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    if not qa_path.exists():
        qa_path.write_text(
            "\n".join(json.dumps(case, ensure_ascii=False) for case in SAMPLE_QA) + "\n",
            encoding="utf-8",
        )
    return raw_dir, qa_path

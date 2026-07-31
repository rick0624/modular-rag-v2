# modular-rag-v2

配置驅動的模組化 RAG 框架 —— **Haystack 2.x 之上的薄層**。

[modular-rag(v1)](https://github.com/rick0624/modular-rag) 是全手刻的
十模組框架;v2 改以 [Haystack](https://haystack.deepset.ai/) 為引擎,
保留 v1 的操作體驗與契約:**槽位式單一 YAML 配置**(換方法只改一行)、
建構期相容性檢查、繁中錯誤訊息(指出收到什麼 / 期望什麼 / 該改哪個
欄位 / 可用替代)、離線可跑的測試文化。自維護程式碼從整套框架縮減為
一個薄 builder(`rag/builder.py`):方法名稱 → Haystack 元件的對映表
加上接線邏輯。

## 架構總覽

```
Ingestion:  Import → Parsing → Chunking → (身分蓋章) → Embedding → Indexing
Inference:  查詢 → Query Transformation 鏈 → 多子查詢檢索(檢索 → 重排)
            → 融合/聚合 → Prompt → Generation
Evaluation: JSONL 測試集 → 逐題查詢 → hit rate / MRR
```

| 槽位 | 方法(粗體為預設) | 對應實作 |
|---|---|---|
| import | **local_file** / pdf_file | 自訂 FileLister(相對路徑 doc_id) |
| parsing | **plain_text** / pdf / clean(鏈用) | Haystack converters + DocumentCleaner |
| chunking | **fixed_size** / structure_based / page_based / no_chunking | (Recursive)DocumentSplitter,一律字元單位 |
| embedding | **mock** / sentence_transformers / api_embedding | ST 整合套件 / 自訂 Flexible API embedder |
| indexing | **in_memory** / elasticsearch | InMemory / Elasticsearch DocumentStore |
| query_transformation | **normalize** / passthrough / glossary / llm_decompose | 自訂元件(`list[str] → list[str]`) |
| retrieval | **bm25** / embedding / hybrid | 依 indexing 選 retriever;hybrid 走 RRF |
| reranking | **none** / similarity / llm | ST cross-encoder / core LLMRanker |
| generation | **mock** / openai / gateway_openai_compatible | OpenAIChatGenerator / 自訂閘道 generator |
| fusion(內建步驟) | rrf / concat_dedup / max_score × group_by none/doc/page | 自訂 SubqueryFusion(v1 演算法) |
| evaluation | basic_retrieval_metrics | hit rate / MRR(doc_id 依名次去重) |

方法組合的相容性(content_type、分頁需求、索引能力)在**建構期**檢查,
不合法組合直接報錯並列出可相容的方法,詳見
[docs/interfaces.md](docs/interfaces.md)。

## 安裝

需要 Python 3.10+。

```bash
pip install -e ".[dev]"        # 核心(離線可跑,無 torch、無 ES)
pip install -e ".[st]"         # 選配:sentence-transformers 模型(embedding / cross-encoder)
pip install -e ".[es]"         # 選配:Elasticsearch
```

## 執行

```bash
python scripts/run_demo.py                                  # 全離線,不需金鑰
python scripts/run_demo.py --config configs/smoke.yaml      # 跑過所有機制(拆解/重排/融合)
python scripts/run_demo.py --config configs/company.yaml    # 公司環境(見下)
```

demo 會自動建立範例語料(`./data/raw`)與評估集(`./data/eval/qa.jsonl`),
依序執行 ingestion → 查詢 → 評估,印出融合後檢索結果、實際送出的
prompt(可稽核)、回答與 hit rate / MRR。

內建三份 config:

| config | 組合 | 適用情境 |
|---|---|---|
| `default.yaml` | mock embedding + in_memory + mock LLM;**同時是方法型錄** | 離線開發、試跑 |
| `smoke.yaml` | 方法鏈 + LLM 拆解(mock 腳本)+ hybrid + LLM 重排(mock 腳本)+ fusion | 煙霧測試 |
| `company.yaml` | ES + 公司 embedding API + 公司 LLM 閘道 + 雙段 rerank + fusion | 公司環境 |

公司環境:

```bash
cp .env.example .env           # 填入 ES_URL 與兩把金鑰(只需第一次)
docker compose up -d           # 或指向既有 ES 叢集
python scripts/run_demo.py --config configs/company.yaml
```

### 測試

```bash
python -m pytest               # 全離線(預設排除 ES 測試)
ES_URL=http://localhost:9200 python -m pytest -m es   # ES 整合測試(選配)
```

### Python API

```python
from rag import build_pipelines, load_config

config = load_config("configs/default.yaml")
pipelines = build_pipelines(config)
pipelines.run_ingestion()
result = pipelines.query("FAISS 支援哪些索引結構?")
result["answer"]      # 回答
result["documents"]   # 融合後切片(meta 含 doc_id/page/group_key/sources)
result["prompt"]      # 實際送出的 prompt(可稽核)
```

## 配置說明

每個槽位固定兩種寫法:`method`(方法名稱)+ `params`(該方法參數),
或 `method_params`(以方法名稱分區,多方法設定並存、切 `method` 即換,
只有被選中的區塊會做驗證):

```yaml
  embedding:
    method: sentence_transformers      # 換方法只改這一行
    method_params:
      sentence_transformers:
        model_name: sentence-transformers/all-MiniLM-L6-v2
      api_embedding:
        endpoint: https://api.example.com/v1/embeddings
```

指定的 method 不存在、參數打錯欄位、組合不相容,都會在**建構期**得到
指出確切位置與可用替代的繁中錯誤訊息。

### 方法鏈(`method` 寫成清單)

輸入輸出同型別的槽位(parsing、query_transformation、reranking)可以
把 `method` 寫成清單依序執行;鏈長 > 1 時參數必須用 `method_params`:

```yaml
  parsing:
    method: [pdf, clean]               # PDF 解析後再清理
  query_transformation:
    method: [normalize, llm_decompose] # 先正規化,再 LLM 拆解子查詢
  reranking:
    method: [similarity, llm]          # cross-encoder 收斂 → LLM 精排
```

### 多子查詢與融合(`fusion`)

`llm_decompose` 把查詢拆成多個子查詢後,每個子查詢**各自檢索與重排**,
再由 fusion 合併;也可用 `group_by` 把結果聚合成文件 / 頁粒度:

```yaml
inference:
  fusion:            # 選填;未設定且單一查詢時 = 傳統直線流程
    group_by: doc    # none(切片)/ doc(按文件)/ page(按頁)
    strategy: rrf    # rrf(名次融合,預設)/ concat_dedup / max_score
    top_k: 5
```

聚合後的切片 meta 帶 `group_key` / `num_merged` / `sources`
(各來源名次與原始分數),診斷可追。

### 環境變數注入(`${ENV_VAR}`)

config 中所有字串支援 `${ENV_VAR}` 佔位符,載入時展開;引用未設定的
變數**直接報錯**並指名變數。機密因此不進版控。變數可放 `.env`
(自動載入;真正的環境變數優先)。要輸出字面值 `${...}` 寫 `$${...}`。

### Escape hatch(原生 Haystack pipeline)

槽位式組不出來的圖形狀,可直接掛原生 pipeline YAML(專家模式,
跳過相容性檢查;與該階段的槽位區塊互斥):

```yaml
haystack_pipelines:
  inference: pipelines/exotic_graph.yaml   # Pipeline.dumps() 的輸出
```

## 如何新增一個自訂方法

三步驟,以「依句號切分查詢的 transform」為例:

**1. 寫一個 Haystack 元件**(或直接用現成元件,跳到第 2 步):

```python
from haystack import component

@component
class BySentenceSplitter:
    """把每條查詢依句號拆成多條。"""

    @component.output_types(queries=list[str])
    def run(self, queries: list[str]):
        out = [s.strip() for q in queries for s in q.split("。") if s.strip()]
        return {"queries": out}
```

**2. 在 `rag/builder.py` 寫 factory 並加進對映表**:

```python
class _BySentenceParams(BaseParams):   # extra="forbid":打錯參數直接報錯
    pass

def _build_by_sentence(raw, ctx):
    _validate_params("query_transformation", "by_sentence", _BySentenceParams, raw)
    return BySentenceSplitter()

TRANSFORM_FACTORIES["by_sentence"] = SlotFactory(build=_build_by_sentence)
```

**3. 在 YAML 改 `method`**:

```yaml
  query_transformation:
    method: by_sentence
```

其他槽位作法相同;有相容性需求時在 `SlotFactory` 上宣告
(`requires_pages`、`required_capabilities` 等),builder 會自動檢查。

## api_embedding 回應形狀對映表

一組參數涵蓋常見的 embedding API 回應結構,欄位對不上時錯誤訊息會
列出回應中實際存在的欄位:

| API 回應結構 | 設定 |
|---|---|
| `{"embeddings": [[...], ...]}` | 預設值即可 |
| `{"result": {"embeddings": [[...], ...]}}` | `embeddings_field: result.embeddings` |
| `{"data": [{"embedding": [...]}, ...]}`(OpenAI 式) | `embeddings_field: data` + `item_field: embedding` |
| `[[...], ...]`(回應本身就是清單,如 HuggingFace TEI) | `embeddings_field: null` |

## 接入實際 LLM

- **`openai`**:官方 OpenAI 或任何相容服務(vLLM、Ollama、Groq…,
  以 `api_base_url` 指定);金鑰預設讀 `OPENAI_API_KEY`。
- **`gateway_openai_compatible`**:為公司內部閘道保留兩個關鍵行為 ——
  `model` 未設定時請求**完全不帶**該欄位(官方 SDK 做不到);
  OpenAI 推理模型(gpt-5 / o 系列)自動忽略 `temperature`、
  改以 `max_completion_tokens` 送出,不需調整 YAML。
- `llm_decompose` 與 `llm` 重排未指定 `generator` 時,**沿用
  generation 槽位的 LLM 設定**(各自新實例);也可各自指定
  (smoke.yaml 用 mock 腳本示範)。

## v1 → v2 遷移註記

| 項目 | v1 | v2 |
|---|---|---|
| 資料物件 | 自訂 pydantic(Chunk 等) | Haystack `Document`,身分欄位在 meta(見 [docs/interfaces.md](docs/interfaces.md)) |
| chunking 參數 | `chunk_size` / `chunk_overlap` | `split_length` / `split_overlap`(仍為字元單位) |
| prompt 模板 | `{context}` / `{query}` | Jinja2:`{{ query }}` + `{% for doc in documents %}`(預設模板已含 `[chunk_id]` 前綴) |
| 非分頁來源的 `page` | `None` | `1`(按文件分組語意等價) |
| reranking `lexical_overlap` | 內建 | 移除;請改用 `similarity`(效果遠佳)或 `llm` |
| `custom_api` importer/parser | 內建 | 未移植(接外部服務請寫自訂元件,見「新增自訂方法」) |
| FAISS 索引 | in_memory_faiss / id_map_faiss | `in_memory`(開發)/ `elasticsearch`(正式);索引持久化交給 ES |
| service 模式(KB 管理 / 藍綠重建) | FastAPI 服務 | 未移植(v2 之後再議;ES alias 可承接大部分需求) |

## 設計要點

- **Haystack 負責**:元件執行、socket 型別驗證、converter / splitter /
  retriever / joiner / ranker / generator 等成熟實作。
- **薄 builder 負責**:槽位 config → pipeline 的翻譯、語意相容性檢查、
  同向量空間紀律(查詢端 embedder 一律派生自 `ingestion.embedding`)、
  繁中錯誤訊息。
- **`Document.id = chunk_id`**(`"{doc_id}::chunk_{seq}"`):同輸入必同
  id,重複 ingest 即 upsert,ES `_id` 穩定。
- **離線優先**:mock embedding / mock LLM 是一級公民;所有測試不碰
  網路,LLM 行為以腳本化 mock 驗證(含 LLMRanker 的 JSON 排序協定)。
- **相依版本**:`haystack-ai>=2.31,<3`;sentence-transformers 元件從
  整合套件 import(2.32 起移出 core);`elasticsearch-haystack>=6.3`
  (ES 8.x)。

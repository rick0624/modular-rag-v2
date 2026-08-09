# 模組功能選項一覽

每個模組(config 槽位)目前可用的方法,以 `rag/methods_ingestion.py` /
`rag/methods_inference.py` 的註冊表(`*_FACTORIES`)為準。換方法 = 改 config 的 `method` 一行;各方法的
參數與範例見 `configs/default.yaml`(方法型錄),輸入輸出契約見
[interfaces.md](interfaces.md)。

標記:🔌 需額外安裝(`pip install -e ".[extra]"`);⛓ 該模組支援方法鏈
(`method: [a, b]` 依序執行)。

## Ingestion

| 模組 | 方法 | 說明 |
|---|---|---|
| 1 Import | `local_file` | 掃描本地資料夾(txt / md / pdf,`extensions` 可收窄) |
| | `custom` | 自訂來源(公司 DMS / API,文件可不落地) |
| 2 Parsing ⛓ | `plain_text` | 純文字檔(txt / md) |
| | `pdf` | PDF(pypdf 文字層 + 選擇性 OCR 🔌 `[ocr]`) |
| | `auto` | 依檔案類型自動分流(混合語料用這個) |
| | `clean` | 清理空行/多餘空白(只能放鏈中,如 `[auto, clean]`) |
| | `custom` | 自訂解析:鏈首 converter(`kind: converter`)或鏈中處理器 |
| 3 Chunking | `fixed_size` | 固定長度切塊(字元數 + 重疊) |
| | `structure_based` | 依結構遞迴切分(頁 → 段 → 行 → 句) |
| | `page_based` | 按頁切(需要會產生頁界的 parser:pdf / auto) |
| | `no_chunking` | 整份文件一個切片 |
| | `custom` | 自訂切塊規則;可生成自訂 meta 欄位並以 `provides_fields` 宣告 |
| 4 Embedding(皆支援 `source_field`:改用 chunking 生成的欄位做向量) | `mock` | 離線確定性偽向量(開發測試用) |
| | `sentence_transformers` | 本地模型 🔌 `[st]` |
| | `api_embedding` | 通用 HTTP embedding API(欄位名可對映,OpenAI 式也用它) |
| 5 Indexing(皆支援 `fields:` 自訂欄位白名單/改名) | `in_memory` | 記憶體索引(隨 process 消失;開發測試用) |
| | `elasticsearch` | ES 索引(向量 + BM25 + filter;支援增量 ingest、自訂 mapping、settings 預建索引與 ingest pipeline)🔌 `[es]` |

## Inference

| 模組 | 方法 | 說明 |
|---|---|---|
| 6 Query Transformation ⛓ | `passthrough` | 原樣通過 |
| | `normalize` | NFKC 正規化、壓空白、轉小寫 |
| | `glossary` | 術語表:命中的定義注入 prompt(不改查詢) |
| | `jargon_mapping` | 術語替換查詢文字(適合檢索-only) |
| | `llm_rewrite` | LLM 查詢改寫(1 → 1) |
| | `llm_decompose` | LLM 查詢拆解(1 → N 子查詢,子查詢各自檢索後融合) |
| | `custom` | 自訂轉換 |
| 7 Retrieval | `bm25` | 關鍵字檢索 |
| | `embedding` | 向量檢索 |
| | `hybrid` | BM25 + 向量,RRF 融合(`boost_k_factor` 可放大候選) |
| | `custom` | 自訂檢索(如公司檢索 API,不吃本地索引) |
| 8 Reranking ⛓ | `none` | 不重排 |
| | `similarity` | cross-encoder 相似度重排 🔌 `[st]` |
| | `api_rerank` | 通用 HTTP rerank API(欄位名可對映) |
| | `llm` | LLM listwise 重排 |
| | `llm_fact_check` | LLM 相關性查核(只過濾,不重排不改分) |
| | `custom` | 自訂重排 |
| 融合/聚合(內建步驟) | (內建) | `strategy: rrf / concat_dedup / max_score` × `group_by: none / doc / page`;內建融合直接寫扁平參數、**不寫 method** |
| | `custom` | 自訂融合(與扁平參數互斥;掛上即一律執行) |
| 9 Generation | `mock` | 離線假答案(可腳本化回覆) |
| | `openai` | 官方 OpenAI(或相容 `api_base_url`:vLLM / Ollama / 代理) |
| | `gateway_openai_compatible` | OpenAI 相容的內部閘道(可不帶 model 欄位) |
| | `custom` | 非 OpenAI 相容的內部推論服務(prompt 仍由框架組) |
| Routing(選填,省略 = 不做) | `keyword_match` | 關鍵字規則分類 |
| | `custom` | 接公司分類模型 / 規則引擎 |
| Formatter(選填,省略 = 不做) | `simple_json` | 通用 JSON 對外格式 |
| | `custom` | 公司信封格式 |

## Evaluation

| 模組 | 方法 | 說明 |
|---|---|---|
| 10 Evaluation(選填,省略 = 不做) | `basic_retrieval_metrics` | 檢索指標 hit_rate / MRR(JSONL 測試集或行內 cases) |

## 備註

- **`custom` 是統一的掛載機制**:每個支援的槽位都可用
  `file` + `class`(或 `class_path`)載入自寫的 Haystack `@component`,
  契約見 interfaces.md §1;範例骨架在 `examples/custom_modules/`,
  完整掛載示範見 `configs/custom_demo.yaml`(inference 端)與
  `configs/custom_ingestion_demo.yaml`(ingestion 端)。
- **LLM 類方法**(`llm_rewrite` / `llm_decompose` / `llm` /
  `llm_fact_check`)都有 `generator` 參數可各自指定 LLM
  (`{method, params}`,吃 Generation 模組的任一方法);
  不設定時沿用 generation 槽位 —— 整條管線因此可以只接一個 LLM 來源。
- **新增方法**:通用功能寫元件 + 在 `methods_ingestion.py` /
  `methods_inference.py` 對應的 `*_FACTORIES` 加一行;公司特定邏輯直接寫 custom module,零框架改動。

# 模組功能選項一覽

每個模組(config 槽位)目前可用的方法,以 `rag/methods_ingestion.py` /
`rag/methods_inference.py` 的註冊表(`*_FACTORIES`)為準。換方法 = 改 config 的 `method` 一行;各方法可設定的
參數見下方[各方法參數](#各方法參數),完整範例見 `configs/default.yaml`
(方法型錄),輸入輸出契約見 [interfaces.md](interfaces.md)。

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
| 4 Embedding(皆支援 `source_field`:改用 chunking 生成的欄位做向量;與 `extra_vectors`:同一模型對額外欄位各出一組向量,寫進 meta 隨索引落地) | `mock` | 離線確定性偽向量(開發測試用) |
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
| | `llm_multi_hyde` | Multi-HyDE:LLM 生成 k 篇多角度假設文件參與檢索(對向量檢索特別有效) |
| | `preqrag` | PreQRAG:先分類 single/multi,單文件問題改寫、跨文件問題拆解(SIGIR 2025 LiveRAG) |
| | `custom` | 自訂轉換 |
| 7 Retrieval | `bm25` | 關鍵字檢索 |
| | `embedding` | 向量檢索 |
| | `hybrid` | BM25 + 向量,RRF 融合(`boost_k_factor` 可放大候選) |
| | `custom` | 自訂檢索(如公司檢索 API,不吃本地索引) |
| 8 Reranking ⛓ | `none` | 不重排 |
| | `similarity` | cross-encoder 相似度重排 🔌 `[st]` |
| | `api_rerank` | 通用 HTTP rerank API(欄位名可對映) |
| | `llm` | LLM listwise 重排 |
| | `insertrank` | InsertRank:listwise 重排時把候選的檢索分數寫進 prompt(WSDM 2026;只重排不改分) |
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

## 各方法參數

每個方法的參數 schema 定義在 `rag/methods_ingestion.py` /
`rag/methods_inference.py` 的 pydantic 模型(`extra="forbid"`:多打或
打錯欄位名建構期直接報錯,並列出可接受的參數)。**本節的預設值以程式
為準**;`configs/default.yaml` 展示的是常用覆寫值,兩者不同時以這裡為準。

參數寫在 config 的兩種位置擇一:`params:`(扁平,單一方法時最簡潔)或
`method_params.<方法名>:`(分區,多方法設定並存、切換 `method` 互不干擾);
方法鏈(`method: [a, b]`)必須用 `method_params` 分區。

標記:**粗體參數為必填**;「共通」列的參數該模組所有內建方法都支援。

### 1 Import

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| `local_file` | **`input_dir`** | — | 要匯入的資料夾路徑 |
| | `extensions` | 全收(`.txt` / `.md` / `.pdf`) | 要納入的副檔名;收窄成同質型別後才能搭配單型別 parser |
| | `recursive` | `true` | 是否遞迴掃描子資料夾 |
| `custom` | [共用 custom 參數](#共用-custom-參數) + `content_type` | `null` | 元件輸出的 content_type 宣告(`text` / `pdf` / `mixed`),供與 parsing 的相容性檢查;`null` = 跳過檢查 |

### 2 Parsing

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| `plain_text` | `encoding` | `utf-8` | 文字檔編碼 |
| `pdf` | `ocr` | `"auto"` | OCR 策略:`off` = 純 pypdf;`auto` = 掃描頁(無文字層)才 OCR;`force` = 全頁 OCR(多欄 / 表格版面順序亂時用)。需 `pip install -e ".[ocr]"` |
| | `ocr_scale` | `2.0` | OCR 前的頁面渲染倍率 |
| `auto` | `encoding` | `utf-8` | 文字 / Markdown 分支的檔案編碼 |
| | `ocr`、`ocr_scale` | 同 `pdf` | pdf 分支的 OCR 參數(同 `pdf` 方法) |
| `clean` | `remove_empty_lines` | `true` | 移除空行 |
| | `remove_extra_whitespaces` | `true` | 壓縮多餘空白 |
| | `remove_repeated_substrings` | `false` | 去除頁首頁尾重複段 |
| `custom` | 共用 custom 參數 + `kind` | `doc_processor` | 鏈位置:`converter` = 鏈首(sources + meta → documents);`doc_processor` = 鏈中 / 鏈尾(documents → documents) |
| | `produces_pages` | `false` | 是否產生頁界資訊(`page_based` chunking 的前提) |
| | `input_content_types` | `null`(全收) | converter 可處理的 content_type 清單;僅 `kind: converter` 可設 |

### 3 Chunking

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| `fixed_size` | `split_length` | `512` | 每個切片的字元數上限 |
| | `split_overlap` | `64` | 相鄰切片重疊的字元數(必須 < `split_length`) |
| `structure_based` | `split_length` | `512` | 每個切片的字元數上限 |
| | `split_overlap` | `0` | 相鄰切片重疊的字元數 |
| | `separators` | `["\f", "\n\n", "\n", "。", " "]` | 遞迴切分的分隔符優先序(頁界 → 段落 → 行 → 句 → 空白) |
| `page_based` | `pages_per_chunk` | `1` | 每個切片包含的頁數 |
| `no_chunking` | (無參數) | | 整份文件一個切片 |
| `custom` | 共用 custom 參數 + `requires_pages` | `false` | 是否需要分頁輸入(供與 parsing 鏈的相容性檢查) |
| | `provides_fields` | `null` | 宣告元件會在切片 meta 生成的欄位;宣告後下游(embedding `source_field` / indexing `fields`)引用錯欄位建構期即報錯。不可用框架保留名 |

### 4 Embedding

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| (共通) | `source_field` | `content` | 文件端 embedding 的輸入欄位:`content` 或 chunking 生成的任一 meta 欄位;查詢端不受影響 |
| | `extra_vectors` | `null` | 額外向量(共用同一個模型):`{向量欄位名: 來源欄位}`,寫進切片 meta 隨索引落地;查詢端內建檢索只用主向量 |
| `mock` | `dim` | `32` | 向量維度 |
| `sentence_transformers` | `model_name` | `sentence-transformers/all-MiniLM-L6-v2` | 模型名稱 |
| `api_embedding` | **`endpoint`** | — | embedding API 端點 URL |
| | `headers` | `{}` | HTTP 標頭(認證放這裡) |
| | `model` | `null` | 模型名稱;`null` 時請求不帶此欄位 |
| | `batch_size` | `16` | 每批送出的文字筆數 |
| | `timeout` | `30.0` | 請求逾時秒數 |
| | `texts_field` | `input` | 請求中放文字的欄位名 |
| | `model_field` | `model` | 請求中放模型的欄位名 |
| | `embeddings_field` | `embeddings` | 回應中向量清單的欄位名(OpenAI 式為 `data`) |
| | `item_field` | `null` | 清單元素內放向量的欄位名(OpenAI 式為 `embedding`) |

### 5 Indexing

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| (共通) | `incremental` | `false` | 增量 ingest:檔案沒變連 parse 都跳過;切片內容沒變跳過 embedding 與寫入 |
| | `fields` | `null`(自訂欄位全帶) | 寫入索引的自訂欄位白名單 + 改名:`{索引欄位名: meta 欄位名}`;框架欄位永遠保留 |
| `in_memory` | (僅共通參數) | | |
| `elasticsearch` | **`hosts`** | — | ES 端點,如 `http://localhost:9200` |
| | `index` | `modular-rag` | 索引名稱 |
| | `api_key` | `null` | ES API key(base64 的 `id:api_key`);與 username/password 擇一 |
| | `username` / `password` | `null` | basic auth,必須成對提供 |
| | `ca_certs` | `null` | CA 憑證路徑(https 叢集用私有 CA 時) |
| | `verify_certs` | `null`(= client 預設 `true`) | 是否驗證伺服器憑證 |
| | `custom_mapping` | `null`(內建 mapping) | 完整覆蓋索引 mapping;須自含 `content` 與 `embedding` 欄位,僅索引不存在時生效 |
| | `settings` | `null` | 建索引時的 index settings(analysis 等);必須搭配 `custom_mapping`,建 pipeline 時就會連線 ES |
| | `ingest_pipeline` | `null` | 寫入時套用的 ES ingest pipeline 名稱(伺服器端須已建立) |
| | `request_timeout` | `null`(= client 預設 10 秒) | 單次請求逾時秒數;大量 bulk 寫入逾時時調高 |
| | `retry_on_timeout` | `null`(= client 預設 `false`) | 逾時是否自動重試 |
| | `max_retries` | `null`(= client 預設 3) | 單次請求的最大重試次數 |

### 6 Query Transformation

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| `passthrough` | (無參數) | | |
| `normalize` | `lowercase` | `true` | 是否把拉丁字母轉為小寫 |
| `glossary` | `glossary` | `null` | 行內術語表(`術語: 定義`);與 `glossary_path` 擇一或並用 |
| | `glossary_path` | `null` | 術語表 YAML 檔路徑 |
| | `expand_query` | `false` | 是否把命中的定義附加到查詢文字(預設只注入 prompt) |
| `jargon_mapping` | `mapping` | `null` | 行內術語對照表(`術語: 直白描述`) |
| | `json_path` | `null` | 對照表 JSON 檔路徑(扁平物件) |
| `llm_rewrite` | `prompt` | 內建改寫 prompt | 需含 `{{ query }}` |
| | `generator` | `null`(沿用 generation 槽位) | 改寫用的 LLM(`{method, params}`,見[備註](#備註)) |
| `llm_decompose` | `max_subqueries` | `4` | 子查詢數上限 |
| | `prompt` | 內建拆解 prompt | 需含 `{{ query }}` |
| | `generator` | `null` | 同上 |
| `llm_multi_hyde` | `num_documents` | `3` | 假設文件篇數 |
| | `keep_original` | `true` | 是否保留原查詢一路(各自檢索後融合) |
| | `prompt` | 內建 HyDE prompt | 需含 `{{ query }}` |
| | `generator` | `null` | 同上 |
| `preqrag` | `num_rewrites` | `2` | single 分支的改寫條數 |
| | `max_subqueries` | `4` | multi 分支的子查詢數上限 |
| | `include_original` | `true` | 是否保留原查詢一路 |
| | `classify_prompt` / `rewrite_prompt` / `decompose_prompt` | 內建 | 各分支 prompt,均需含 `{{ query }}` |
| | `generator` | `null` | 分類 / 改寫 / 拆解共用 |
| `custom` | 共用 custom 參數 | | |

### 7 Retrieval

`bm25` / `embedding` / `hybrid` 參數相同:

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| `bm25` / `embedding` / `hybrid` | `top_k` | `10` | 取回的切片數上限 |
| | `boost_k_factor` | `1` | 候選放大倍率:各 retriever 取回 `top_k × boost_k_factor` 筆,供下游 rerank 收斂 |
| `custom` | 共用 custom 參數 | | 外部檢索(不吃本地索引);`top_k` 等由 `init_params` 透傳 |

### 8 Reranking

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| `none` | (無參數) | | |
| `similarity` | `model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder 模型;中文語料請換多語模型 |
| | `top_k` | `10` | 重排後保留筆數 |
| `api_rerank` | **`endpoint`** | — | rerank API 端點(完整 URL) |
| | `headers` | `{}` | HTTP 標頭(認證放這裡) |
| | `model` | `null` | 模型名稱;`null` 時請求不帶該欄位 |
| | `top_k` | `5` | 重排後保留筆數 |
| | `timeout` | `30.0` | 請求逾時秒數 |
| | `query_field` | `question` | 請求中放查詢的欄位名 |
| | `documents_field` | `documents` | 請求中放候選文字的欄位名 |
| | `model_field` | `model` | 請求中放模型的欄位名 |
| | `results_field` | `returnData` | 回應中結果清單的欄位(支援 `a.b` 巢狀;回應本身是清單時設 `null`) |
| | `index_field` | `index` | 結果元素中名次索引的欄位名 |
| | `score_field` | `score` | 結果元素中分數的欄位名 |
| | `index_base` | `0` | 回應 index 的起算基準(`0` 或 `1`;設錯會整體位移一格) |
| | `higher_is_better` | `true` | 分數越大越相關;回傳「距離」的 API 設 `false` |
| | `raise_on_failure` | `false` | API 失敗時中斷查詢;預設保留原檢索順序並記警告 |
| `llm` | `top_k` | `5` | 重排後保留筆數 |
| | `generator` | `null`(沿用 generation 槽位) | 重排用的 LLM(`{method, params}`) |
| `insertrank` | `top_k` | `5` | 重排後保留筆數 |
| | `score_label` | `檢索分數` | prompt 中分數的名稱,依上游檢索器據實描述 |
| | `prompt` | 內建重排 prompt | 需含 `{{ query }}` 與 `{{ documents }}` |
| | `generator` | `null` | 同上 |
| `llm_fact_check` | `prompt` | 內建查核 prompt | 需含 `{{ query }}` 與 `{{ documents }}` |
| | `max_docs` | `null`(全部送查) | 送交 LLM 查核的切片數上限;其餘原樣通過 |
| | `generator` | `null` | 同上 |
| `custom` | 共用 custom 參數 | | |

### 融合 / 聚合(fusion)

fusion 區塊不走 `method` / `method_params` 形狀:內建融合直接寫扁平參數
(**不寫 method**),custom 融合寫 `method: custom` + `params`,兩者互斥。

| 寫法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| 內建 | `strategy` | `rrf` | 融合策略:`rrf` / `concat_dedup` / `max_score` |
| | `group_by` | `none` | 聚合鍵:`none` / `doc` / `page` |
| | `top_k` | `5` | 融合後保留的筆數上限 |
| `method: custom` | `params` | | 共用 custom 參數(`file` + `class` 或 `class_path`、`init_params`);掛上即一律執行 |

### 9 Generation

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| (共通) | `prompt_template` | `null`(內建模板) | Jinja2 模板,可用 `{{ query }}`、`{{ documents }}`、`{{ glossary_notes }}` |
| | `system_prompt` | `null` | system 角色訊息 |
| `mock` | `replies` | `null`(可辨識的假答案) | 腳本化回覆,依序循環 |
| `openai` | `model` | `gpt-5-mini` | OpenAI 模型名稱 |
| | `api_key` | `null`(用 `OPENAI_API_KEY` 環境變數) | API key |
| | `api_base_url` | `null` | 替代 base URL(vLLM / Ollama / 代理閘道) |
| | `temperature` | `null`(不帶此欄位) | 取樣溫度;gpt-5 系列不接受非預設值 |
| | `max_tokens` | `null` | 回覆長度上限 |
| | `timeout` | `null` | 請求逾時秒數 |
| `gateway_openai_compatible` | **`base_url`** | — | 閘道 base URL,如 `https://llm.example.com/v1` |
| | `api_key` | `null` | API key;建議用 `${ENV_VAR}` 注入 |
| | `model` | `null`(請求不帶 model 欄位) | 模型名稱 |
| | `temperature` / `max_tokens` | `null` | 同 `openai` |
| | `timeout` | `60.0` | 請求逾時秒數 |
| | `headers` | `{}` | 額外 HTTP 標頭 |
| | `completions_path` | `/chat/completions` | completions 端點路徑 |
| `custom` | 共用 custom 參數 + 共通 prompt 參數 | | prompt 仍由框架組裝,模板設定與內建方法同款 |

### Routing(選填槽位)

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| `keyword_match` | **`routes`** | — | 類別 → 關鍵字清單(查詢包含關鍵字即命中) |
| | `default_category` | `general` | 無任何命中時回傳的類別 |
| `custom` | 共用 custom 參數 | | |

### Formatter(選填槽位)

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| `simple_json` | `include_content` | `true` | payload 是否包含切片內文(`false` = 只留引用資訊) |
| `custom` | 共用 custom 參數 | | |

### 10 Evaluation

| 方法 | 參數 | 預設 | 說明 |
|---|---|---|---|
| `basic_retrieval_metrics` | `dataset_path` | `null` | JSONL 測試集路徑;與 `cases` 擇一 |
| | `cases` | `null` | 行內測試案例清單(`{query, relevant_doc_ids}`) |

### 共用 custom 參數

所有槽位的 `method: custom` 都吃同一組定位參數(`rag/custom.py` 的
`CustomModuleParams`);個別槽位的額外宣告欄位(import 的 `content_type`、
parsing 的 `kind` 等)見各槽位表格。

| 參數 | 預設 | 說明 |
|---|---|---|
| `class_path` | `null` | 元件類別的 import 路徑(`pkg.mod:ClassName`);與 `file` **恰好擇一** |
| `file` | `null` | 元件 `.py` 檔路徑(相對執行目錄);需搭配 `class` |
| `class` | `null` | `file` 檔案中的類別名稱(僅搭配 `file`) |
| `init_params` | `{}` | 透傳給元件建構子的參數 |

## 備註

- **`custom` 是統一的掛載機制**:每個支援的槽位都可用
  `file` + `class`(或 `class_path`)載入自寫的 Haystack `@component`,
  契約見 interfaces.md §1;範例骨架在 `examples/custom_modules/`,
  完整掛載示範見 `configs/custom_demo.yaml`(inference 端)與
  `configs/custom_ingestion_demo.yaml`(ingestion 端)。
- **LLM 類方法**(`llm_rewrite` / `llm_decompose` / `llm_multi_hyde` /
  `preqrag` / `llm` / `insertrank` / `llm_fact_check`)都有 `generator`
  參數可各自指定 LLM
  (`{method, params}`,吃 Generation 模組的任一方法);
  不設定時沿用 generation 槽位 —— 整條管線因此可以只接一個 LLM 來源。
- **新增方法**:通用功能寫元件 + 在 `methods_ingestion.py` /
  `methods_inference.py` 對應的 `*_FACTORIES` 加一行;公司特定邏輯直接寫 custom module,零框架改動。

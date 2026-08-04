# 介面契約(Haystack 語境)

v2 不再自訂資料物件:模組之間傳遞的是 Haystack 的 `Document` 與
`ChatMessage`。框架的契約因此縮減成兩件事:**Document 的 meta 鍵**
(哪些欄位一定存在、語意是什麼)+ **槽位的輸入輸出形狀**。
只要這兩者不變,方法怎麼換、元件內部怎麼改,都不影響其他部分。

## 1. Document meta 鍵契約

切片(經過 `ChunkMetaStamper` 之後的 Document)保證帶有:

| meta 鍵 | 型別 | 語意 |
|---|---|---|
| `doc_id` | str | 來源文件識別碼 = 檔案相對 `input_dir` 的 POSIX 路徑;**跨執行穩定**(同一來源永遠同 id),upsert / 評估都靠它 |
| `seq` | int | 文件內切片序號(0 起,全文件連續,不分頁重排) |
| `page` | int \| None | 來源頁碼(1 起)。分頁來源(PDF)為實際頁碼;**非分頁來源為 1**(v1 為 None;按文件分組的語意等價) |
| `chunk_id` | str | `"{doc_id}::chunk_{seq}"`;**同時也是 `Document.id`** |
| `source` / `importer` | str | 來源路徑與匯入方法(由 FileLister 提供,經 converter 流動) |

融合 / 聚合後(`SubqueryFusion` 輸出)額外帶有:

| meta 鍵 | 語意 |
|---|---|
| `group_key` | 聚合鍵:`chunk_id` / `doc_id` / `"{doc_id}#p{page}"` |
| `num_merged` | 此組合併的來源筆數 |
| `sources` | 各來源的 `{subquery_index, rank, score}`(重排後名次與原始分數) |

**`Document.id = chunk_id` 是刻意的設計**(偏離 Haystack 的內容雜湊
預設):id 穩定 → `DocumentWriter(policy=OVERWRITE)` 即 upsert 語意、
ES 的 `_id` 可預測。代價:切片內容在 stamper 之後**不得再改動**
(id 不會跟著變)。

## 2. 不變量(修改程式時不可破壞)

- **分數**:一律越大越相關;結果依分數降冪;只在同一次結果內可比
  (cosine、BM25、RRF 量綱不同,不可跨方法比較)。
- **識別碼**:`doc_id` / `chunk_id` 確定性 —— 同輸入必同 id。
- **空白切片不產生輸出**(頁碼由 splitter 先數好,不受過濾影響)。
- **同向量空間**:查詢端 embedder 一律由 `ingestion.embedding` 派生
  (沒有 inference 端 embedding 槽位),結構上保證與索引同源。
  換 embedding 方法或模型後必須重建索引。
- **prompt 可稽核**:`RagPipelines.query()` 回傳的 `prompt` 就是實際
  送給 LLM 的內容;切片一律帶 `[chunk_id]` 前綴,引用可回溯。

## 3. 槽位輸入輸出(Haystack 語境)

```
Ingestion:  import(FileLister) [→ SourceChangeFilter(檔案層增量)]
            → sources+meta → parsing(converter[→processor…];pdf 走
            PdfToDocument = pypdf 文字層 + 選擇性 OCR)
            → documents → chunking(splitter) → ChunkMetaStamper
            [→ IncrementalChangeFilter(切片層增量)]
            → embedding(document embedder) → DocumentWriter(store)

Inference:  query:str → query_transformation 鏈(list[str] → list[str])
            → MultiQueryRetrievalStage(內部:retrieval [→ reranking 鏈])
            → list[list[Document]] → SubqueryFusion → list[Document]
            → ChatPromptBuilder → chat generator → replies
            (generate_answer: false 時圖止於 SubqueryFusion,檢索-only)
            └→ routing(選填獨立支線):query:str → route: dict
               吃「原始」查詢(不經 transform 鏈)、圖上無下游;
               結果附加於 query() 回傳的 routing key 與 trace,不影響檢索

Evaluation: JSONL 測試集 → 逐題 RagPipelines.query() → hit_rate / MRR
```

| 槽位 | 輸入 → 輸出 | 方法鏈 |
|---|---|---|
| import | (params)→ `sources: list[str]` + `meta: list[dict]` | ✗ |
| parsing | sources → `documents`(鏈首 converter,其餘 Document→Document)。鏈首可展開為**內部圖**(`auto`:FileTypeRouter → 多 converter → DocumentJoiner),對外契約不變 | ✓ |
| chunking | documents → documents(splitter;`no_chunking` = 無節點) | ✗ |
| embedding | factory 一次建 (document_embedder, text_embedder) 一對 | ✗ |
| indexing | factory 回傳 document store(+ 能力宣告) | ✗ |
| query_transformation | `queries: list[str]` → `list[str]`(glossary 另輸出 `notes`;jargon_mapping **替換**查詢文字,glossary 只附註) | ✓ |
| retrieval | 展開為內部圖(retriever[s] + joiner),輸出 documents;`boost_k_factor` 放大候選為 top_k × factor | ✗ |
| reranking | documents(+query)→ documents;**只能重排/過濾/改分,不得改內容**(llm_fact_check 純過濾:順序與分數皆保留) | ✓ |
| generation | ChatPromptBuilder 的 messages → `replies: [ChatMessage]`;`generate_answer: false` 時整段省略(此時 generation 區塊選填,僅作 LLM 沿用來源) | ✗ |
| routing(選填槽位) | `query: str` → `route: dict[str, Any]`(圖上無下游;route 內容自由,慣例含 `category`) | ✗ |
| fusion(內建步驟) | `list[list[Document]]` → `list[Document]`(非槽位,`inference.fusion` 設定) | — |

## 4. 相容性宣告(建構期檢查)

方法在 factory 表上宣告、`rag/compatibility.py` 在建構期檢查,
不合法組合直接報錯並列出可相容替代:

1. **content_type**:import 宣告 `output_content_type`(靜態)或
   `output_content_type_fn`(動態:`local_file` 依 `extensions` 推導
   —— 同質 → `text` / `pdf`,異質 → `mixed`);parsing 鏈首宣告
   `input_content_types`(`auto` 接受全部三種)。
2. **pages**:chunking 宣告 `requires_pages`,parsing 鏈任一環節
   `produces_pages` 即滿足。
3. **索引能力**:indexing 宣告 `capabilities`(`vector_search` /
   `text_search` / `metadata_filter` / `incremental_update`),
   retrieval 宣告 `required_capabilities`,需為子集。
   `indexing.params.incremental: true`(增量 ingest,兩層:importer 後的
   `SourceChangeFilter` 按檔案雜湊跳過未變檔案的 parse;stamper 後的
   `IncrementalChangeFilter` 按 chunk_id 比對內容跳過 embedding)需要
   `incremental_update` 能力。檔案 manifest 帶 parse 設定雜湊,設定
   變更即作廢(全量重 parse)。

新增相容性維度:在 `SlotFactory` 加宣告欄位、`compatibility.py`
補一條檢查即可。

## 4b. custom module 契約(`rag/contracts.py` + `rag/custom.py`)

`method: custom` 把使用者自寫的 Haystack `@component` 掛進槽位
(config 以 `class_path: "pkg.mod:Class"` 或 `file` + `class` 指定,
`init_params` 透傳建構子)。與 §4 的分工:§4 檢查「方法組合的語意」
(宣告式欄位),本節檢查「元件本身的 socket 形狀」(introspection)——
custom 元件無宣告可查,建構期直接檢視
`__haystack_input__` / `__haystack_output__`。

支援的槽位與契約(`SLOT_CONTRACTS`):

| 槽位 | 必要輸入 sockets | 必要輸出 sockets |
|---|---|---|
| query_transformation | `queries: list[str]` | `queries: list[str]` |
| retrieval | `query: str` | `documents: list[Document]` |
| reranking | `query: str`、`documents: list[Document]` | `documents: list[Document]` |
| routing | `query: str` | `route: dict[str, Any]` |

驗證規則(不符 → `ConfigError`,訊息指明缺什麼、實際有什麼、怎麼改):

- 契約 sockets 必須存在,型別採**寬鬆相容**(與 `Pipeline.connect`
  同判準,私有 API 失效時退回型別名稱比對)。
- **額外輸出 sockets 允許**(`query()` 以 `include_outputs_from` 收集
  全節點輸出,額外輸出自動進 trace)。
- **契約外的必填輸入不允許**(圖上沒有上游會餵它,執行期必缺輸入);
  有預設值的額外輸入允許。
- **邊界映射責任**:外部系統的欄位在 custom 元件**內部**轉成 canonical
  型別(內文 → `Document.content`、分數 → `score`、其餘無損進 `meta`),
  並建議補 §1 的 `doc_id` / `chunk_id` 契約鍵。槽位之間永遠只流
  canonical 型別 —— 讓公司格式流出邊界,就是模組間的兩兩耦合。
- 其他槽位(generation / embedding / indexing …)的 factory 回傳形狀
  不是單一元件,暫不支援 custom;需求出現時再為其設計契約。

範例骨架:`examples/custom_modules/` + `configs/custom_demo.yaml`
(有整合測試 `tests/test_custom_demo.py` 保證永遠可跑)。

## 5. 服務模式不變量(`rag/service.py` + `rag/kb_meta.py`)

- **索引內容必須與 ingestion 設定一致**。強制機制:ingestion 指紋 =
  對「展開前」的原始 config dict 取 `{ingestion, haystack_pipelines.ingestion}`
  區塊,`json.dumps(sort_keys=True)` 後 sha256。展開前計算 →
  `${ENV_VAR}` 機密不進雜湊;解析後的 dict → 註解 / 排版 / anchor
  重構不影響。已知盲區:env var **值**的輪替、escape-hatch pipeline
  檔的內容變更、custom module **.py 檔內容**的變更(config 中的
  file / class_path 路徑有進指紋,檔案內容沒有;inference 端的 custom
  以 `/reload` 重載即可生效 —— 每次建構都重新 exec 檔案)。
- 指紋**排除操作性欄位**:`indexing` 的 `incremental` 與連線憑證 / TLS
  (`api_key` / `username` / `password` / `ca_certs` / `verify_certs`)——
  它們決定「怎麼跑」與「連不連得上」,不決定索引裡有什麼;索引位置仍由
  `hosts` 與 `index` 決定(改了就會偵測到)。
- 儲存位置:`elasticsearch` → index mapping `_meta`(跟索引走);
  其他 → store 物件屬性(單 process 內有效)。
- `/reload` 只重建 inference(store 沿用);指紋不符 → **409**,
  導向 `/ingest`。**`/ingest` 是 ingestion 設定變更的唯一通道**
  (全新 store、兩條 pipeline 全重建、成功後才更新指紋)。
- 單 process、單 worker;pipeline 執行與狀態切換以同一把 lock 序列化。

## 6. 新需求進來時怎麼判斷

1. **是「同一槽位的另一種做法」嗎?**(換 LLM、新切法、新指標)
   → 通用的做法:寫元件 + factory + 對映表加一行(進框架型錄);
   專案 / 公司特定的邏輯:寫成 custom module(`method: custom`,
   零框架改動,見 §4b)。九成需求在這裡,見 README「新增自訂方法」。
2. **是「資料要多帶一點資訊」嗎?**(新的追溯欄位)
   → 加 meta 鍵:只加不改,舊元件不讀新鍵也不會壞。
3. **是「新的檔案型別」嗎?**(docx、html…)
   → `builder.py` 的 `_EXTENSION_CONTENT_TYPES` 加副檔名對映 +
   `auto` 的 ParsingGraph 加一條 router 分支(mime type + converter)。
4. **真的是「流程本身多一個步驟」嗎?**
   → 先想能不能作為既有槽位的方法或 fusion 類內建步驟;真的要改圖,
   改 `builder.py` 的組裝邏輯(這正是薄 builder 存在的意義);
   極端形狀走 `haystack_pipelines` escape hatch。

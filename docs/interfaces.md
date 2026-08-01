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
Ingestion:  import(FileLister) → sources+meta → parsing(converter[→processor…])
            → documents → chunking(splitter) → ChunkMetaStamper
            → embedding(document embedder) → DocumentWriter(store)

Inference:  query:str → query_transformation 鏈(list[str] → list[str])
            → MultiQueryRetrievalStage(內部:retrieval [→ reranking 鏈])
            → list[list[Document]] → SubqueryFusion → list[Document]
            → ChatPromptBuilder → chat generator → replies
            (generate_answer: false 時圖止於 SubqueryFusion,檢索-only)

Evaluation: JSONL 測試集 → 逐題 RagPipelines.query() → hit_rate / MRR
```

| 槽位 | 輸入 → 輸出 | 方法鏈 |
|---|---|---|
| import | (params)→ `sources: list[str]` + `meta: list[dict]` | ✗ |
| parsing | sources → `documents`(鏈首 converter,其餘 Document→Document) | ✓ |
| chunking | documents → documents(splitter;`no_chunking` = 無節點) | ✗ |
| embedding | factory 一次建 (document_embedder, text_embedder) 一對 | ✗ |
| indexing | factory 回傳 document store(+ 能力宣告) | ✗ |
| query_transformation | `queries: list[str]` → `list[str]`(glossary 另輸出 `notes`;jargon_mapping **替換**查詢文字,glossary 只附註) | ✓ |
| retrieval | 展開為內部圖(retriever[s] + joiner),輸出 documents;`boost_k_factor` 放大候選為 top_k × factor | ✗ |
| reranking | documents(+query)→ documents;**只能重排/過濾/改分,不得改內容**(llm_fact_check 純過濾:順序與分數皆保留) | ✓ |
| generation | ChatPromptBuilder 的 messages → `replies: [ChatMessage]`;`generate_answer: false` 時整段省略(此時 generation 區塊選填,僅作 LLM 沿用來源) | ✗ |
| fusion(內建步驟) | `list[list[Document]]` → `list[Document]`(非槽位,`inference.fusion` 設定) | — |

## 4. 相容性宣告(建構期檢查)

方法在 factory 表上宣告、`rag/compatibility.py` 在建構期檢查,
不合法組合直接報錯並列出可相容替代:

1. **content_type**:import 宣告 `output_content_type`,parsing 鏈首
   宣告 `input_content_types`。
2. **pages**:chunking 宣告 `requires_pages`,parsing 鏈任一環節
   `produces_pages` 即滿足。
3. **索引能力**:indexing 宣告 `capabilities`(`vector_search` /
   `text_search` / `metadata_filter` / `incremental_update`),
   retrieval 宣告 `required_capabilities`,需為子集。

新增相容性維度:在 `SlotFactory` 加宣告欄位、`compatibility.py`
補一條檢查即可。

## 5. 新需求進來時怎麼判斷

1. **是「同一槽位的另一種做法」嗎?**(換 LLM、新切法、新指標)
   → 新增方法:寫元件(或直接用 Haystack 現成元件)+ factory +
   對映表加一行。九成需求在這裡,見 README「新增自訂方法」。
2. **是「資料要多帶一點資訊」嗎?**(新的追溯欄位)
   → 加 meta 鍵:只加不改,舊元件不讀新鍵也不會壞。
3. **真的是「流程本身多一個步驟」嗎?**
   → 先想能不能作為既有槽位的方法或 fusion 類內建步驟;真的要改圖,
   改 `builder.py` 的組裝邏輯(這正是薄 builder 存在的意義);
   極端形狀走 `haystack_pipelines` escape hatch。

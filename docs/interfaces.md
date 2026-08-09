# 介面契約

模組之間傳遞的是 Haystack 的 `Document` 與 `ChatMessage`,不自訂資料物件。
契約 = **每個模組的輸入輸出形狀** + **Document 的 meta 鍵**;兩者不變,
方法怎麼換都不影響其他模組。可用的方法選項見 [methods.md](methods.md)。

## 1. 模組 Input / Output 一覽

| 模組 | Input | Output |
|---|---|---|
| 1 Import | (無;來源寫在 params) | `sources: list[str \| Path \| ByteStream]` + `meta: list[dict]`(每筆必帶 `doc_id`) |
| 2 Parsing | `sources` + `meta` | `list[Document]`(內容變純文字;鏈首 converter 之後的鏈中方法為 Document → Document) |
| 3 Chunking | `list[Document]` | `list[Document]`(切片;meta 逐塊複製) |
| 4 Embedding | 建索引端:`list[Document]`;查詢端:`str` | 建索引端:`list[Document]`(帶 `embedding` 向量);查詢端:向量(兩端同一方法派生,保證同向量空間) |
| 5 Indexing | 寫入:`list[Document]`(含向量) | document store(檢索由 Retrieval 模組查它;向量或文字,可帶 filters) |
| 6 Query Transformation | `queries: list[str]` | `queries: list[str]`(1 筆 = 不拆解,即傳統流程;可方法鏈) |
| 7 Retrieval | `query: str` | `documents: list[Document]` |
| 8 Reranking | `query: str` + `documents: list[Document]` | `documents: list[Document]`(只能重排/過濾/改分,不得改內容;可方法鏈) |
| 融合/聚合(pipeline 內建步驟,非獨立模組) | `results: list[list[Document]]` | `documents: list[Document]` |
| 9 Generation | `messages: list[ChatMessage]`(prompt 由框架的 ChatPromptBuilder 組好) | `replies: list[ChatMessage]`(至少一則,取 `replies[0]`) |
| 10 Evaluation | `list[EvalCase]` + 各題 `query()` 完整輸出 | metrics dict(`hit_rate` / `mrr` + 逐題明細) |
| Routing(選填支線) | `query: str`(原始查詢,不經 transform) | `route: dict`(不影響檢索,附加於 `query()` 輸出) |
| Formatter(選填終端支線) | `documents` + `query: str` | `payload: Any`(進 `query()` 輸出的 `output` 鍵) |

custom module(`method: custom`)掛進槽位時,元件的 socket 必須符合上表
同一行的形狀 —— 建構期以 introspection 驗證,不符直接報錯並指明缺什麼。

## 2. Document meta 鍵

切片(經 `ChunkMetaStamper` 後)保證帶有:

| meta 鍵 | 語意 |
|---|---|
| `doc_id` | 來源文件識別碼(檔案相對 `input_dir` 的路徑);跨執行穩定,upsert / 評估靠它 |
| `seq` | 文件內切片序號(0 起) |
| `page` | 來源頁碼(1 起;非分頁來源為 1) |
| `chunk_id` | `"{doc_id}::chunk_{seq}"`;**同時是 `Document.id`**(→ 寫入即 upsert;代價:切片內容此後不得改動) |

融合後另帶 `group_key`(聚合鍵)、`num_merged`(合併筆數)、`sources`
(各來源的子查詢/名次/分數)。

custom chunker 可額外生成自訂 meta 欄位(建議以 `provides_fields` 宣告,
供建構期檢查),下游可引用:embedding 的 `source_field` 選它做向量、
indexing 的 `fields` 決定寫入索引的欄位與名稱。自訂欄位不可使用框架
保留名(doc_id/seq/page/chunk_id/content/embedding/id)。

## 3. 不變量(修改程式時不可破壞)

- **分數**:越大越相關、結果降冪;只在同一次結果內可比(不可跨方法比較)。
- **識別碼確定性**:同輸入必同 `doc_id` / `chunk_id`。
- **同向量空間**:查詢端 embedder 由 `ingestion.embedding` 派生,
  換 embedding 方法或模型後必須重建索引。
- **prompt 可稽核**:`query()` 回傳的 `prompt` 即實際送 LLM 的內容,
  切片帶 `[chunk_id]` 前綴,引用可回溯。
- **邊界映射責任**:外部系統的欄位在 custom 元件內部轉成 canonical 型別
  (內文 → `Document.content`、分數 → `score`、其餘進 `meta`);
  模組之間永遠只流 canonical 型別。

## 4. 建構期檢查

不合法組合在建 pipeline 時直接報錯(不會跑到一半才炸):

1. **content_type**:import 宣告輸出型別(text / pdf / mixed),
   parsing 鏈首宣告可接受的型別。
2. **pages**:`page_based` chunking 需要會產生頁界的 parser(pdf / auto)。
3. **索引能力**:retrieval 所需能力(向量/文字/filter/增量)必須是
   indexing 方法宣告能力的子集。
4. **custom socket**:custom 元件的輸入輸出 socket 必須符合 §1 契約;
   額外輸出允許(自動進 trace),契約外的必填輸入不允許。
5. **欄位引用**:chunking 宣告了 `provides_fields` 時,embedding 的
   `source_field` 與 indexing 的 `fields` 引用未宣告欄位即報錯;
   `incremental: true` 且被 embed 的自訂欄位不在 `fields` 白名單時
   也報錯(欄位須寫入索引才能在下次 ingest 比對)。

## 5. 服務模式要點

- **索引內容必須與 ingestion 設定一致**:ingestion 區塊(含 custom .py
  檔內容)取雜湊為指紋,存於索引旁;`/reload` 指紋不符 → 409,
  ingestion 設定變更一律走 `/ingest`(全量重建)。
- `/reload` 只重建 inference(索引沿用);inference 端 custom 檔案改動
  重載即生效。

## 6. 新需求怎麼接

1. 同一模組的另一種做法 → 通用的進框架型錄(元件 + factory 一行);
   公司特定的寫 custom module(零框架改動)。九成需求在這裡。
2. 資料要多帶資訊 → 加 meta 鍵(只加不改)。
3. 新檔案型別 → 副檔名對映 + `auto` 加一條分流。
4. 流程真的要多一步 → 改 `rag/builder.py` 的組裝邏輯。

# modular-rag-v2

配置驅動的模組化 RAG 框架 —— **Haystack 2.x 之上的薄層**。

[modular-rag(v1)](https://github.com/rick0624/modular-rag) 是全手刻的
十模組框架;v2 改以 [Haystack](https://haystack.deepset.ai/) 為引擎,
保留 v1 的操作體驗與契約:**槽位式單一 YAML 配置**(換方法只改一行)、
建構期相容性檢查、繁中錯誤訊息(指出收到什麼 / 期望什麼 / 該改哪個
欄位 / 可用替代)、離線可跑的測試文化。自維護程式碼從整套框架縮減為
一個薄 builder:方法型錄(`rag/methods_ingestion.py` /
`rag/methods_inference.py`,方法名稱 → Haystack 元件)加上接線邏輯
(`rag/builder.py`)。

## 架構總覽

```
Ingestion:  Import → Parsing → Chunking → (身分蓋章) → Embedding → Indexing
Inference:  查詢 → Query Transformation 鏈 → 多子查詢檢索(檢索 → 重排)
            → 融合/聚合 ─┬→ Prompt → Generation
            │            └→ Formatter(選填終端支線:對外格式,進輸出的 output key)
            └→ Routing(選填獨立支線:查詢分類,結果附加於輸出)
Evaluation: JSONL 測試集 → 逐題查詢 → hit rate / MRR
```

| 槽位 | 方法(粗體為預設) | 對應實作 |
|---|---|---|
| import | **local_file**(萬用:txt/md/pdf,`extensions` 可收窄) / custom | 自訂 FileLister(相對路徑 doc_id)/ 自訂元件(公司 DMS / API) |
| parsing | **auto**(依檔案類型分流) / plain_text / pdf / clean(鏈用) / custom(鏈首或鏈中,`kind` 宣告);pdf 支援 `ocr: off/auto/force` | FileTypeRouter + 自訂 PdfToDocument(pypdf + rapidocr)+ DocumentCleaner |
| chunking | **fixed_size** / structure_based / page_based / no_chunking / custom | (Recursive)DocumentSplitter,一律字元單位 / 自訂元件(公司切塊規則) |
| embedding | **mock** / sentence_transformers / api_embedding;皆支援 `source_field`(選任一 chunking 生成欄位做向量)與 `extra_vectors`(同一模型對額外欄位各出一組向量) | ST 整合套件 / 自訂 Flexible API embedder |
| indexing | **in_memory** / elasticsearch(皆支援 `incremental: true` 增量 ingest 與 `fields:` 欄位白名單/改名;ES 另支援 custom_mapping + settings 預建索引) | InMemory / Elasticsearch DocumentStore |
| query_transformation | **normalize** / passthrough / glossary / jargon_mapping / llm_rewrite / llm_decompose / llm_multi_hyde / preqrag / custom | 自訂元件(`list[str] → list[str]`) |
| retrieval | **bm25** / embedding / hybrid(皆支援 `boost_k_factor` 候選放大) / custom | 依 indexing 選 retriever;hybrid 走 RRF |
| reranking | **none** / similarity / api_rerank / llm / insertrank / llm_fact_check / custom | ST cross-encoder / 自訂 Flexible API ranker / core LLMRanker / 自訂 InsertRankLLMRanker / 自訂 LLMFactChecker |
| generation | **mock** / openai / gateway_openai_compatible / custom(`generate_answer: false` 可跳過) | OpenAIChatGenerator / 自訂閘道 generator / 自訂元件(`messages → replies`) |
| routing(選填槽位,省略=不做) | keyword_match / custom | 自訂 KeywordRouteClassifier;結果進 `query()` 的 `routing` key,不影響檢索 |
| formatter(選填槽位,省略=不做) | simple_json / custom | 終端支線:最終結果組成對外格式,進 `query()` 的 `output` key;canonical 鍵照舊 |
| fusion(內建步驟,可換 custom) | rrf / concat_dedup / max_score × group_by none/doc/page,或 `method: custom` | 自訂 SubqueryFusion(v1 演算法)/ 自訂元件(掛上即一律執行) |
| evaluation | basic_retrieval_metrics | hit rate / MRR(doc_id 依名次去重) |

`custom` 方法讓你把**自己的 Haystack 元件 .py 檔**掛進槽位,零框架改動
接入公司系統(見下方「如何新增一個自訂方法」與
[configs/custom_demo.yaml](configs/custom_demo.yaml))。

方法組合的相容性(content_type、分頁需求、索引能力)在**建構期**檢查,
不合法組合直接報錯並列出可相容的方法,詳見
[docs/interfaces.md](docs/interfaces.md)。

## 安裝

需要 Python 3.10+。

```bash
pip install -e ".[dev]"        # 核心(離線可跑,無 torch、無 ES)
pip install -e ".[st]"         # 選配:sentence-transformers 模型(embedding / cross-encoder)
pip install -e ".[es]"         # 選配:Elasticsearch
pip install -e ".[service]"    # 選配:HTTP 服務模式(FastAPI + uvicorn)
pip install -e ".[ocr]"        # 選配:PDF OCR(掃描檔、多欄表格版面)
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

### 分階段執行(`--stage`)

兩支腳本都吃 `--stage`,決定這個 process 做哪一段。**只決定建不建那條
pipeline,不改變任何一條的組法**,所以半條與整條的行為完全一致:

| `--stage` | `run_demo.py` | `serve.py` |
|---|---|---|
| `all`(預設) | ingestion → 查詢 → 評估 | 啟動時 ingest 一次,再開服務 |
| `ingestion` | 只建索引(寫 ingestion 指紋),不查詢也不評估 | 只建索引就結束,**不開 port** |
| `inference` | 只查詢 + 評估,索引沿用既有內容 | 跳過啟動 ingestion,直接開服務 |

```bash
# 建索引(排程 / CI)→ 查詢端另外起,兩者靠 ingestion 指紋握手
python scripts/serve.py   --config configs/docs.yaml --stage ingestion
python scripts/serve.py   --config configs/docs.yaml --stage inference

# 只測 inference:改 prompt / 改寫 / 重排的迭代不必每次重切塊 + 重算 embedding
python scripts/run_demo.py --config configs/docs.yaml --stage inference --query "你的問題"
```

`--stage inference` 的前提是**索引裡已經有內容**:

| 情形 | 行為 |
|---|---|
| `indexing: elasticsearch`(索引先前建好) | 比對 ingestion 指紋,不符會出聲(設定變了但索引沒重建 → 結果會無聲劣化);`serve.py` 直接拒絕啟動 |
| `retrieval: custom`(外部檢索,不吃本地索引) | 直接跑 |
| `indexing: in_memory` 且 retrieval 走本地索引 | 索引是空的 → 印警告,查詢不會有結果 |

程式接入時對應 `build_pipelines(config, stage=...)`:沒建的那條在
`RagPipelines` 上是 `None`,誤呼叫 `run_ingestion()` / `query()` 會直接
報錯並指明要怎麼重建。服務端對應 `create_app(path, stage=...)` 與
`rag.service.ingest_only(path)`。

### 逐步紀錄(trace)

終端機預設只印重點(索引筆數、檢索結果、評估摘要、log 檔路徑);
要看**中間每一步做了什麼**加 `--trace`:

```bash
python scripts/run_demo.py --config configs/condense.yaml --trace
python scripts/run_demo.py --config configs/condense.yaml --trace --trace-docs 0   # 每步印出全部切片
python scripts/run_demo.py --config configs/condense.yaml --log-level DEBUG        # 另含 LLM 實際 prompt 與回覆
```

會逐步列出:ingestion 各步驟的產出筆數(import → parse → chunk → stamp →
embed → write)、query transformation 每一段改寫後的查詢、每條子查詢在
**各路檢索器**與**每段重排**後的切片與分數(含該步移除了哪些切片)、
以及 fusion 的最終結果。

資料來源是 `query()` 回傳的 `trace` 欄位與 `run_ingestion()` 的
`trace`,程式接入時可直接取用,不必解析 CLI 輸出:

```python
result = pipelines.query("...")
for step in result["trace"]["subqueries"][0]["steps"]:
    print(step["component"], step["type"], len(step["documents"]))
```

### Log 檔

**每次執行都會寫一份完整紀錄**到 `logs/run-<時間戳>.log`,不需要任何旗標;
路徑會印在執行結束時。內容比終端機多:

| | 終端機 | log 檔 |
|---|---|---|
| 層級 | `--log-level`(預設 WARNING) | 一律 DEBUG |
| 逐步紀錄 | 只在 `--trace` 時 | 一律完整 |
| 每步切片 | 依 `--trace-docs` 截斷(預設 5) | 全印,不截斷 |
| LLM 的 prompt 與回覆 | 需 `--log-level DEBUG` | 一律記錄 |
| HTTP 請求、Haystack 元件執行順序 | ✗ | ✓ |

```bash
python scripts/run_demo.py --config configs/condense.yaml       # 自動寫 logs/run-*.log
python scripts/run_demo.py --log-file logs/experiment-a.log     # 指定路徑
python scripts/run_demo.py --no-log-file                        # 不寫檔
```

log 檔是 UTF-8。Windows PowerShell 5.1 的 `Get-Content` 預設用系統 ANSI 編碼
讀檔,中文會變亂碼,要加旗標:

```powershell
Get-Content logs\run-20260802-154249.log -Encoding UTF8
```

**每個步驟不論成敗都有紀錄**:pipeline 每一步執行完都寫一行 INFO
(元件名稱 + 產出筆數,一律進 log 檔);fail-soft 機制(API 掛掉保留原
順序、LLM 故障退回原查詢…)降級時記 WARNING,訊息帶 `fail-soft:` 前綴
(`grep fail-soft logs/run-*.log` 可一次找出所有降級)。執行結束時終端機
會總結:

```
⚠ 本次執行有 1 則警告(可能含 fail-soft 降級 —— 流程完成但部分步驟已退化):
  - rag.components.api_clients: fail-soft:rerank API 失敗(...),保留原檢索順序前 5 筆
```

沒有這行就代表全程沒有任何降級 ——「跑完了」和「每一步都真的成功」
從此分得開。未捕捉的例外也會連 traceback 寫進 log 檔(不再只進 stderr)。

程式接入用 `rag.logging_config.setup_logging(log_file, console_level)`;
執行後可用 `rag.logging_config.warning_tally()` 取得本次全部警告;
排版函式在 `rag.trace`(`format_ingestion_trace` / `format_query_trace`),
終端機與 log 檔共用同一份實作。

內建三份 config:

| config | 組合 | 適用情境 |
|---|---|---|
| `default.yaml` | mock embedding + in_memory + mock LLM;**同時是方法型錄** | 離線開發、試跑 |
| `smoke.yaml` | 方法鏈 + LLM 拆解(mock 腳本)+ hybrid + LLM 重排(mock 腳本)+ fusion | 煙霧測試 |
| `company.yaml` | ES + 公司 embedding API + 公司 LLM 閘道 + 雙段 rerank + fusion | 公司環境 |
| `condense.yaml` | query condense(術語替換 + LLM 改寫)+ hybrid 候選放大 + rerank + LLM 事實查核 + top 3,**不做生成** | 檢索-only 服務 |
| `docs.yaml` | 同 `condense.yaml`,但來源是混合文件(txt/md/pdf,auto 分流) | 測試自己的文件 |

### 用自己的文件(txt / md / pdf 混放)

把文件放進 **`./data/docs/`**(txt / md / pdf 混放皆可,子資料夾也會掃到),然後:

```bash
python scripts/run_demo.py --config configs/docs.yaml --query "你的問題"
```

`local_file` 是萬用 importer(預設收 `.txt` / `.md` / `.pdf`),搭配
`parsing: auto` 依檔案類型自動分流(txt/md → 文字 converter、pdf →
PyPDFToDocument),全部進**同一個索引**(一個 config = 一個 KB)。

content_type 由 `extensions` 推導:同質(全文字或全 PDF)→ 該型別,
可搭配單型別 parser(`plain_text` / `pdf`);異質 → `mixed`,只有 `auto`
能接。所以**要用 `plain_text` 就得釘住 `extensions: [".txt", ".md"]`**,
否則建構期會報不相容(訊息會提示改用 `auto`)。

> 遷移註記:`pdf_file` import 方法已移除 —— 改用 `local_file`
> (需要純 PDF 語意時加 `extensions: [".pdf"]`)。

幾個常會想調的地方:

- **OCR**(`parsing.params.ocr`,需 `pip install -e ".[ocr]"`):
  - `auto`(預設):掃描檔(無文字層)的頁面自動 OCR;沒裝 OCR 時會
    明確警告該檔「沒有產出任何切片」,不再靜默消失。
  - `force`:全頁 OCR。**多欄 / 表格版面的 PDF(如榜單)必用** ——
    pypdf 按內部串流順序抽字,視覺上相鄰的標題與內容會被打散重排,
    OCR 按視覺位置輸出才正確。成本約每頁數秒,搭配 `incremental: true`
    只有第一次(或檔案變更時)要付。
- **切法**:`chunking.method` 可改 `page_based`(按頁切;混合語料下
  txt/md 整檔視為一頁)或 `structure_based`;版面雜訊多時
  `parsing.method` 改成鏈 `[auto, clean]`。
- **索引**:`docs.yaml` 用獨立索引 `modular-rag-docs`,不會跟範例語料混在一起。
  換 embedding 模型後必須換索引名或先刪索引(ES 的 `dense_vector` dims 建立後不可變)。
- **要生成答案**:把 `generate_answer` 改成 `true`。
- **評估**:`docs.yaml` 刻意不設 `evaluation` —— 內建的 `qa.jsonl` 是給範例語料的。
  自備 JSONL 時 `doc_id` 就是檔案相對 `input_dir` 的路徑(例如 `manual.pdf`)。

公司環境:

```bash
cp .env.example .env           # 填入 ES_URL 與兩把金鑰(只需第一次)
docker compose up -d           # 或指向既有 ES 叢集
python scripts/run_demo.py --config configs/company.yaml
```

#### 連到有認證的 Elasticsearch

`docker-compose.yml` 的本地 ES 關掉了 security,所以不用帶憑證;正式 /
公司叢集預設是開著的,沒帶憑證時第一個請求就會失敗:

```
AuthenticationException(401, 'security_exception',
  'missing authentication credentials for REST request [/]')
```

這不是設定錯誤,而是**還沒設定認證**。在 `indexing.params` 補上一組
(兩者擇一,機密一律用 `${ENV_VAR}` 從 `.env` 注入):

```yaml
indexing:
  method: elasticsearch
  params:
    hosts: ${ES_URL}
    index: modular-rag-company
    username: ${ES_USERNAME}     # basic auth,需與 password 成對
    password: ${ES_PASSWORD}
    # api_key: ${ES_API_KEY}     # 或改用 API key(base64 的 "id:api_key")
    # ca_certs: ${ES_CA_CERTS}   # https 且憑證由私有 CA 簽發時
```

補完再跑一次即可。相關細節:

- **`username` / `password` 必須成對**,且不能與 `api_key` 併用 —— 兩種
  情況都在建 pipeline 時就報錯,不會拖到送出請求才失敗。
- 憑證也可以直接寫在 URL 裡(`ES_URL=https://user:pass@es.example.com:9200`),
  client 認得;但密碼含特殊字元時要先 URL-encode,且 URL 常被記進 log,
  仍建議用上面的欄位。
- 換成 401 以外的錯誤代表認證已經通了:`403` 是帳號權限不足(需要目標索引的
  `create_index` / `read` / `write`),連線階段的 TLS 錯誤則用 `ca_certs`
  指到公司 CA 憑證解決。

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
result["routing"]     # 查詢分類結果(未設 routing 槽位時為 None)
```

## 服務模式(HTTP API)

`run_demo.py` 是批次模式:跑完一題就結束。長駐服務改用 `serve.py` ——
**一份 config 啟動一個 KB 服務**,ingestion 與 inference 同駐一個 process
(共用 store,embedding 等跨階段設定結構上保證一致),啟動時 ingest 一次,
之後查詢不再重建索引:

```bash
pip install -e ".[service]"
python scripts/serve.py --config configs/docs.yaml                    # 啟動並 ingest
python scripts/serve.py --config configs/docs.yaml --stage inference  # 索引已建好(會驗指紋)
python scripts/serve.py --config configs/docs.yaml --stage ingestion  # 只建索引就結束(不開 port)
```

讀寫分離部署:`--stage ingestion` 由排程 / CI 建索引,查詢服務用
`--stage inference` 起來吃同一個 ES 索引,兩者靠 ingestion 指紋握手
(見上方「分階段執行」)。

端點(另有 `/docs` 自動 API 文件):

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "混合檢索怎麼運作?"}'
curl -X POST http://127.0.0.1:8000/reload   # 改了 YAML 的 inference 段後套用
curl -X POST http://127.0.0.1:8000/ingest   # 資料或 ingestion 設定變更後全量重建
```

**啟動後只允許改 inference / evaluation 區塊。** ingestion 區塊
(import / parsing / chunking / embedding / indexing)決定索引裡的
內容,改了就必須重建索引 —— 服務以 **ingestion 指紋**強制這件事:
ingest 時把 ingestion 區塊的 sha256(以展開前的原始 config 計算,
機密不進雜湊)寫在索引旁(ES:index mapping `_meta`;in_memory:
process 內),`/reload` 與 `--stage inference` 啟動時比對,不符即拒絕
(409 / 拒啟)並導向 `POST /ingest`。

注意事項:

- **增量 ingest**(`indexing.params.incremental: true`,`docs.yaml` 已開),
  兩層:**檔案層** —— 內容未變的檔案連 parse(含 OCR)都跳過(檔案
  bytes 雜湊記在索引旁的 manifest;parsing/chunking 設定變更會使 manifest
  作廢、全量重 parse);**切片層** —— 變更檔案中「content 與被 embed 的
  欄位(`source_field`)」皆未變的切片跳過 embedding。注意:只改
  `indexing.params.fields` 映射時,增量不會回填已跳過的切片,請重建索引
  或關 incremental 全量跑一次。回應的 `skipped_files` / `skipped_unchanged` 顯示兩層跳過量;
  `documents_written: 0` 表示這次沒有任何變更。
- **空來源回報**:沒有產出任何切片的檔案(掃描檔沒 OCR、解析失敗…)
  會出現在回應的 `empty_sources` 與 log 警告中,不會靜默消失。
- **單 worker**:store 與服務狀態都在 process 內,查詢 / 重建以 lock
  序列化;不要用 `uvicorn --workers N`。
- **in_memory 索引**:`/reload` 在 process 內沒問題,但重啟即失、
  `--stage inference` 起來是空索引;正式服務請用 `elasticsearch`。
- **重 ingest 不會刪除已移除檔案的舊切片**(upsert 語意,增量模式亦同):
  來源檔案刪減後想要乾淨的索引,換索引名或先刪索引再 `/ingest`。

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

## 如何新增一個自訂方法

兩條路:**custom module**(路 A,推薦 —— 零框架改動,程式碼留在你自己
的 repo)與**進框架型錄**(路 B —— 改方法型錄檔,成為所有人可選的方法)。

### 路 A:custom module(`method: custom`)

**1. 寫一個 Haystack 元件 .py 檔**(放哪都行,例:`my_transform.py`):

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

**2. 在 YAML 掛進槽位**:

```yaml
  query_transformation:
    method: custom
    params:
      file: ./my_transform.py       # 路徑相對「執行目錄」,不是 config 檔位置
      class: BySentenceSplitter
      # init_params: {...}          # 透傳給元件建構子
```

已安裝成套件的元件改用 `class_path: "my_pkg.transforms:BySentenceSplitter"`
(與 `file` 擇一,不需要 `class`)。

支援 custom 的槽位與 **socket 契約**(建構期驗證,不符直接報錯並指明修法):

| 槽位 | 輸入 sockets | 輸出 sockets |
|---|---|---|
| query_transformation | `queries: list[str]` | `queries: list[str]` |
| retrieval | `query: str` | `documents: list[Document]` |
| reranking | `query: str` + `documents: list[Document]` | `documents: list[Document]` |
| routing | `query: str` | `route: dict[str, Any]` |
| generation | `messages: list[ChatMessage]` | `replies: list[ChatMessage]` |
| fusion | `results: list[list[Document]]` | `documents: list[Document]`(建議另輸出 `applied: bool`) |
| import | (無 —— ingestion 以 `pipeline.run({})` 啟動) | `sources: list[str \| Path \| ByteStream]` + `meta: list[dict]`(**每筆必帶 `doc_id`**) |
| parsing(鏈首,`kind: converter`) | `sources` + `meta`(同 import 輸出) | `documents: list[Document]` |
| parsing(鏈中,`kind: doc_processor`,預設) | `documents: list[Document]` | `documents: list[Document]` |
| chunking | `documents: list[Document]` | `documents: list[Document]`(meta 逐塊複製,`doc_id` 必須保留) |
| formatter | `documents: list[Document]` + `query: str` | `payload: Any`(**終端槽位特權**:型別自由,元件仍須宣告實際的具體型別) |

規則與慣例:

- **額外的輸出 socket 允許**(自動收進 trace);**額外的必填輸入不允許**
  (執行期沒有上游會餵它)—— 需要設定值就給預設值或走 `init_params`。
- **外部欄位映射進 Document 是元件內部的責任**:外部系統的回應
  (如 `{dockey, score, contentTitle, contentChunk}`)在 custom retrieval
  內轉成 `Document`(內文 → `content`、分數 → `score`、其餘 → `meta`),
  出了槽位邊界只流框架的 canonical 型別。建議一併補 `meta["doc_id"]` 與
  `meta["chunk_id"]`,trace 標籤、fusion `group_by: doc` 與 evaluation
  才能直接運作。
- custom 可出現在方法鏈中(`method: [normalize, custom]`),但同一條鏈
  只能有一個 `custom`(需要兩個時把邏輯合併成一個元件,或走路 B)。
- **log 要看得到**:`logging.getLogger(__name__)` 在 `file:` 載入時可用
  (模組掛在框架設成 DEBUG 的 `_rag_custom.*` 底下),但 `class_path:`
  載入時模組名是你自己的套件路徑,層級會繼承 root(`--log-level`,預設
  WARNING)—— INFO / DEBUG 在**發出當下**就被丟掉,連 log 檔的 DEBUG
  handler 都輪不到。兩種載入方式都想要紀錄,就把 logger 命名在 `rag.*`
  底下:`logging.getLogger("rag.custom.my_transform")`(範例骨架都是這樣
  寫的)。這樣終端機仍只印 WARNING 以上,細節照樣進 log 檔。
- **generation 的 custom 是「換掉 LLM 客戶端」,不是換掉 prompt 組裝**:
  契約就是 Haystack 的 ChatGenerator 形狀,`prompt_template` /
  `system_prompt` 照常寫在 YAML,元件收到的是框架組好的 messages。
  同一支元件也能掛在 `llm_rewrite` / `llm_decompose` / `llm_multi_hyde` /
  `preqrag` / `llm` / `insertrank` / `llm_fact_check` 的
  `params.generator`,整條 pipeline 只走公司的推論服務。OpenAI 相容的閘道**不需要**寫 custom,用
  `gateway_openai_compatible` 即可。
- **fusion 掛了 custom 就一律執行**:單一查詢(N=1)也進元件,「單查詢
  原樣通過」的內建行為不會幫你做,由元件自己決定。建議額外輸出
  `applied: bool` 供 trace 區分;跨子查詢的原始分數不可直接比較,
  安全預設是名次法(RRF)。與內建參數(group_by / strategy / top_k)
  互斥,混用會在載入時報錯。
- **formatter 的信封固定、內容自由**:槽位位置(fusion 之後的終端支線,
  與 prompt → generation 並聯)、socket 名稱(`payload`)與 `query()` 的
  鍵(`output`)固定,payload 的型別由元件自己決定(dict / str / 自訂
  類別)—— 這是終端槽位的特權,圖上沒有下游接它;中間槽位不可模仿。
  走 HTTP(`/query` 回應的 `output` 欄位)時 payload 必須 JSON 可序列化。
- **ingestion 端的相容性宣告寫在 config 參數裡**:import 的
  `content_type`、parsing 的 `kind` / `produces_pages` /
  `input_content_types`、chunking 的 `requires_pages` —— 建構期的相容性
  檢查(content_type 流向、分頁需求)照常執行,省略即跳過 / 用預設。
- **ingestion 端 custom 的 `file:` 檔案內容會進 ingestion 指紋**:改了
  解析 / 切塊邏輯 = 索引內容過期,服務模式 `/reload` 會 409,請走
  `POST /ingest` 重建;`incremental: true` 的檔案層 manifest 也會作廢
  (全量重 parse)。`class_path:` 指向已安裝套件,只有路徑字串進指紋,
  內容變更偵測不到 —— 要指紋保護就用 `file:`。另注意 `--stage inference`
  的查詢端主機也要有同一份 `.py`,否則指紋對不上。
- **custom import 回傳非本地路徑(ByteStream / API 參照)時**,
  `incremental: true` 的檔案層增量無從比對雜湊,退化為每次全量重 parse
  (會出聲警告;切片層增量仍會跳過內容未變的 embedding)。
- embedding / indexing 槽位暫不支援 custom(factory 回傳形狀不是單一
  元件:embedding 是 document / text 一對,indexing 是 document store);
  有需求時走路 B。
- 完整可跑的骨架見 [examples/custom_modules/](examples/custom_modules/)
  (改寫 / 檢索 / 重排 / 分類 / 生成 / 融合 / 匯入 / 解析 / 切塊 / 格式化,
  `TODO(替換點)` 標明換入真實邏輯的位置)與兩份示範 config:
  `python scripts/run_demo.py --config configs/custom_demo.yaml --trace`
  (inference 端 + fusion;custom retrieval 不吃本地索引,也能加
  `--stage inference` 只跑查詢)、
  `python scripts/run_demo.py --config configs/custom_ingestion_demo.yaml --trace`
  (ingestion 三槽位:公司 API 匯入 → 自訂解析 → 自訂切塊)
- 注意:custom module 就是執行任意 Python 程式碼,與 config 檔同一
  信任層級,只載入你信任的來源。

### 路 B:進框架型錄(改 `rag/methods_inference.py` 或 `rag/methods_ingestion.py`)

**1. 寫元件**(同上)。**2. 寫 factory 並加進對映表**(inference 端方法
改 `rag/methods_inference.py`,ingestion 端改 `rag/methods_ingestion.py`):

```python
class _BySentenceParams(BaseParams):   # extra="forbid":打錯參數直接報錯
    pass

def _build_by_sentence(raw, ctx):
    validate_params("query_transformation", "by_sentence", _BySentenceParams, raw)
    return BySentenceSplitter()

TRANSFORM_FACTORIES["by_sentence"] = SlotFactory(build=_build_by_sentence)
```

(`BaseParams` / `validate_params` / `SlotFactory` 都在 `rag/slots.py`。)

**3. 在 YAML 改 `method: by_sentence`**。

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

## api_rerank 請求 / 回應形狀對映表

同樣的作法用在 rerank API 上。預設值對應的形狀是請求
`{"question", "documents", "model"}`、回應 `{"returnData": [{"index", "score"}]}`:

| API 形狀 | 設定 |
|---|---|
| 請求 `{"question", "documents", "model"}` | 預設值即可 |
| 請求 `{"query", "documents"}` | `query_field: query` |
| 回應 `{"returnData": [{"index", "score"}]}` | 預設值即可 |
| 回應 `{"results": [{"index", "relevance_score"}]}`(Cohere 式) | `results_field: results` + `score_field: relevance_score` |
| 回應 `[{"index", "score"}]`(回應本身就是清單) | `results_field: null` |
| 回應的 `index` 從 1 起算 | `index_base: 1` |
| 分數是「距離」(越小越相關) | `higher_is_better: false` |

行為上要知道的幾件事:

- **`index` 指的是候選在送出清單中的位置**,不是文件 id。`index_base`
  設錯會讓結果整體位移一格 —— 全部越界時會直接報錯並提示改設哪個值,
  但只差一格而沒越界時偵測不到,接線時請先確認 API 文件。
- **回應未列出的候選視同淘汰**,與 `llm` 重排的協定一致。API 端若有
  自己的 top_n,最終筆數會是它與 `top_k` 取小。
- **不自動分批**:rerank 的分數只在同一次呼叫內可比,分批送會讓排序
  失真。候選數由 `retrieval` 的 `top_k` × `boost_k_factor` 決定,
  API 有長度上限時請從那裡收斂。
- **fail-soft**:API 掛掉或回應解析不了時記 WARNING 並保留原檢索順序
  (前 `top_k` 筆),查詢不中斷。初次接線建議先設 `raise_on_failure: true`
  把欄位對映確認好再關掉。
- **多子查詢會多次呼叫**:`llm_decompose` 拆成 N 個子查詢時,每個子查詢
  各跑一次重排 → N 次 API 呼叫(`similarity` / `llm` 也是如此)。

## 接入實際 LLM

- **`openai`**:官方 OpenAI 或任何相容服務(vLLM、Ollama、Groq…,
  以 `api_base_url` 指定);金鑰預設讀 `OPENAI_API_KEY`。
- **`gateway_openai_compatible`**:為公司內部閘道保留兩個關鍵行為 ——
  `model` 未設定時請求**完全不帶**該欄位(官方 SDK 做不到);
  OpenAI 推理模型(gpt-5 / o 系列)自動忽略 `temperature`、
  改以 `max_completion_tokens` 送出,不需調整 YAML。
- `llm_decompose`、`llm_rewrite`、`llm_multi_hyde`、`preqrag`、`llm` 重排、
  `insertrank` 與 `llm_fact_check` 未指定 `generator` 時,**沿用 generation
  槽位的 LLM 設定**(各自新實例);也可各自指定(smoke.yaml 用 mock
  腳本示範)。

## 檢索-only 模式(`generate_answer: false`)

系統只需要檢索結果、不需要 LLM 生成答案時,在 `inference` 設定
`generate_answer: false`:

- pipeline 止於 fusion(去重、排序、裁 `top_k`),不建 prompt_builder
  與 generator;`query()` 回傳的 key 不變,但 `answer` / `prompt` /
  `reply_meta` 為 `None`,檢索結果照常在 `documents`。
- `generation` 區塊此時**可省略**;保留時只作為 `llm_rewrite` /
  `llm` / `llm_fact_check` 等 LLM 方法的沿用連線來源(見上)。
  兩者都沒有時,這些方法必須各自指定 `params.generator`。
- 完整範例見 `configs/condense.yaml`:術語替換(`jargon_mapping`,
  JSON 對照表)→ LLM 改寫(`llm_rewrite`)→ hybrid 檢索候選放大
  (`boost_k_factor: 3`,各 retriever 取 top_k × 3)→ cross-encoder
  收斂 → LLM 事實查核(`llm_fact_check`,只移除不相關切片,不重排
  不改分)→ fusion 去重排序回傳 top 3。

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
| service 模式(KB 管理 / 藍綠重建) | FastAPI 服務 | 已移植:`scripts/serve.py`(單 KB;ingestion 指紋守護設定一致性,見「服務模式」節) |

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

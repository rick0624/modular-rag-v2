"""KB 中繼資料:ingestion 設定指紋的計算與存取。

服務模式的核心不變量 —— **索引內容必須與 ingestion 設定一致**
(embedding 模型、切塊參數…都會影響索引裡的東西)。指紋把這個不變量
變成可機械檢查的:ingest 時把指紋寫進索引旁,之後 /reload 或
--stage inference 啟動時比對,「設定變了但索引沒重建」在第一時間被擋下,
而不是等到查詢結果悄悄劣化。

指紋的計算基礎是 **展開前** 的原始 config dict
(:func:`rag.config.load_raw_config`):

- ``${ENV_VAR}`` 佔位符原樣保留 → 機密永不進雜湊;
- 解析後的 dict → 註解、排版、鍵順序、YAML anchor 重構都不影響指紋;
- 已知限制:環境變數「值」的輪替,以及 escape hatch
  (``haystack_pipelines.ingestion``)所指 pipeline 檔的內容變更,
  指紋看不見(只雜湊路徑字串)。

儲存位置依 indexing 方法而異:

- ``elasticsearch``:索引 mapping 的 ``_meta``(跟著索引走,跨 process
  / 重啟都在;不用特殊文件存 —— 會污染檢索結果)。
- 其他(``in_memory``):store 物件屬性(單 process 服務內有效;
  重啟即失,與 in_memory 資料本身的生命週期一致)。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FINGERPRINT_KEY = "modular_rag_ingestion_fingerprint"
_STORE_ATTR = "_modular_rag_ingestion_fingerprint"

MANIFEST_KEY = "modular_rag_source_manifest"
_MANIFEST_ATTR = "_modular_rag_source_manifest"


_OPERATIONAL_INDEXING_KEYS = (
    "incremental",
    # 連線憑證與 TLS 設定:決定「連不連得上」,不決定索引裡有什麼。
    # 補上認證(或換一組)不該讓既有索引被判定為過期。
    "api_key",
    "username",
    "password",
    "ca_certs",
    "verify_certs",
)


def _strip_operational_keys(ingestion: Any) -> Any:
    """移除不影響索引「內容」的操作旗標,避免指紋誤報。

    ``indexing.params.incremental``(與 ``method_params.*.incremental``)
    只決定 ingest 的執行方式(跳不跳過未變切片),開關它不會讓索引內容
    與設定不一致 —— 不該觸發「請重建索引」。ES 的認證 / TLS 欄位同理:
    只影響連線本身,索引的位置仍由 ``hosts`` 與 ``index`` 決定。
    """
    if not isinstance(ingestion, dict):
        return ingestion
    result = json.loads(json.dumps(ingestion, default=str))  # 深拷貝
    indexing = result.get("indexing")
    if isinstance(indexing, dict):
        params = indexing.get("params")
        if isinstance(params, dict):
            for key in _OPERATIONAL_INDEXING_KEYS:
                params.pop(key, None)
        method_params = indexing.get("method_params")
        if isinstance(method_params, dict):
            for block in method_params.values():
                if isinstance(block, dict):
                    for key in _OPERATIONAL_INDEXING_KEYS:
                        block.pop(key, None)
    return result


# ingestion 端支援 custom module 的槽位:這些槽位的 ``file:`` **檔案內容**
# 要進指紋 —— 改了解析 / 切塊邏輯沒重建索引,索引內容就與設定不一致,
# 與改 chunking 參數同罪。inference 端的 custom 刻意不雜湊(改了走 /reload
# 即生效,不影響索引內容);class_path 指向已安裝套件,無單一檔案可雜湊,
# 只有路徑字串進指紋(文件註明此限制)。
_CUSTOM_FILE_SLOTS = ("import", "parsing", "chunking")


def custom_file_hashes(ingestion: Any) -> dict[str, str]:
    """收集 ingestion 端 custom module 的 ``file:`` 內容雜湊。

    輸入為原始(或解析後)的 ingestion dict;對 :data:`_CUSTOM_FILE_SLOTS`
    中 method 含 "custom" 的槽位,依 ``params_for`` 的優先序
    (``method_params.custom`` 蓋過 ``params``)取 ``file`` 並雜湊其內容。
    檔案讀不到(路徑含未展開的 ``${ENV_VAR}``、或建索引的主機上沒這個檔)
    時該槽位記 WARNING 並退回 path-only —— 指紋仍含路徑字串,
    只是看不見內容變更。
    """
    hashes: dict[str, str] = {}
    if not isinstance(ingestion, dict):
        return hashes
    for slot in _CUSTOM_FILE_SLOTS:
        block = ingestion.get(slot)
        if not isinstance(block, dict):
            continue
        method = block.get("method")
        methods = method if isinstance(method, list) else [method]
        if "custom" not in methods:
            continue
        method_params = block.get("method_params")
        custom_block = (
            method_params.get("custom") if isinstance(method_params, dict) else None
        )
        params = custom_block if isinstance(custom_block, dict) else block.get("params")
        file = params.get("file") if isinstance(params, dict) else None
        if not isinstance(file, str):
            continue
        try:
            digest = hashlib.sha256(Path(file).read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            # path-only 退路:路徑字串本來就在指紋裡,但檔案內容的變更
            # 從此偵測不到 —— 必須出聲,否則改了切塊邏輯不會被要求重建。
            logger.warning(
                "fail-soft:讀不到 %s 槽位的 custom 檔案 '%s'(%s),"
                "指紋退回 path-only(該檔內容變更將偵測不到)",
                slot, file, exc,
            )
            continue
        hashes[slot] = digest
    return hashes


def ingestion_fingerprint(raw_config: dict[str, Any]) -> str:
    """計算 ingestion 相關區塊的 sha256 指紋(輸入為展開前的原始 dict)。"""
    ingestion = raw_config.get("ingestion")
    payload = {
        "ingestion": _strip_operational_keys(ingestion),
        "haystack_pipelines.ingestion": (
            (raw_config.get("haystack_pipelines") or {}).get("ingestion")
        ),
    }
    # 非空才併入:沒用 ingestion custom 的 config 指紋逐位元不變(相容)。
    file_hashes = custom_file_hashes(ingestion)
    if file_hashes:
        payload["custom_module_files"] = file_hashes
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,  # YAML 原生型別(日期等)不是 JSON 可序列化,以字串表示
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _es_read_meta(store: Any, index_name: str | None) -> dict[str, Any]:
    """讀取 ES index mapping 的整個 ``_meta``;索引不存在時回傳空 dict。"""
    from elasticsearch import NotFoundError

    try:
        mapping = store.client.indices.get_mapping(index=index_name)
    except NotFoundError:  # 索引還不存在 = 從未 ingest 過
        logger.debug("索引 %s 不存在,視為從未 ingest", index_name)
        return {}
    return mapping[index_name]["mappings"].get("_meta") or {}


def _es_merge_meta(store: Any, index_name: str | None, key: str, value: Any) -> None:
    """合併寫入單一鍵:ES 的 ``_meta`` 是整包替換,直接寫會蓋掉其他鍵。"""
    merged = {**_es_read_meta(store, index_name), key: value}
    store.client.indices.put_mapping(index=index_name, meta=merged)


def read_fingerprint(
    indexing_method: str, store: Any, index_name: str | None = None
) -> str | None:
    """讀取索引旁的指紋;不存在時回傳 None。"""
    if indexing_method == "elasticsearch":
        return _es_read_meta(store, index_name).get(FINGERPRINT_KEY)
    return getattr(store, _STORE_ATTR, None)


def write_fingerprint(
    indexing_method: str, store: Any, fingerprint: str, index_name: str | None = None
) -> None:
    """把指紋寫到索引旁(ES:mapping ``_meta``;其他:store 屬性)。"""
    if indexing_method == "elasticsearch":
        _es_merge_meta(store, index_name, FINGERPRINT_KEY, fingerprint)
        return
    setattr(store, _STORE_ATTR, fingerprint)


# --- 來源檔案 manifest(檔案層增量:沒變的檔案連 parse 都跳過)---------
#
# 形狀:{"key": <parse 設定雜湊>, "files": {doc_id: 檔案 sha256}}。
# ``key`` 是護欄:parsing / chunking / OCR 等設定變了,同一份檔案會產出
# 不同切片,舊 manifest 必須整份作廢(否則檔案沒變就跳過 parse,索引
# 內容會停留在舊設定)。


def parse_config_key(ingestion_dump: dict[str, Any]) -> str:
    """由解析後的 ingestion 設定計算 manifest 的作廢鍵。

    custom module 的 ``file:`` 內容一併納入:custom parsing / chunking 的
    .py 改了,同一份檔案會產出不同切片 —— 只看 config 的話檔案層增量會
    誤判「沒變就跳過 parse」,索引停留在舊解析邏輯。
    """
    payload: dict[str, Any] = dict(ingestion_dump)
    file_hashes = custom_file_hashes(ingestion_dump)
    if file_hashes:  # 非空才併入:既有 manifest 的 key 不變
        payload["__custom_module_files__"] = file_hashes
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_manifest(
    indexing_method: str, store: Any, index_name: str | None = None
) -> dict[str, Any] | None:
    """讀取來源 manifest;不存在時回傳 None。"""
    if indexing_method == "elasticsearch":
        return _es_read_meta(store, index_name).get(MANIFEST_KEY)
    return getattr(store, _MANIFEST_ATTR, None)


def write_manifest(
    indexing_method: str,
    store: Any,
    manifest: dict[str, Any],
    index_name: str | None = None,
) -> None:
    """把來源 manifest 寫到索引旁(與指紋同機制,合併寫入)。"""
    if indexing_method == "elasticsearch":
        _es_merge_meta(store, index_name, MANIFEST_KEY, manifest)
        return
    setattr(store, _MANIFEST_ATTR, manifest)


def indexing_info(config: Any) -> tuple[str, str | None]:
    """回傳 config 的 (indexing 方法, ES index 名稱或 None);中繼資料存取用。"""
    ing = getattr(config, "ingestion", None)
    if ing is None:  # escape hatch:原生 pipeline,無槽位資訊
        return "", None
    method = ing.indexing.methods()[0]
    return method, ing.indexing.params_for(method).get("index")

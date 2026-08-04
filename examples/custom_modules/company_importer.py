"""公司 import custom module 骨架(佔位範例,邏輯待替換)。

``import`` 槽位的 custom module 範本:從公司系統(DMS / wiki / API)
拉文件進 ingestion,取代掃本地資料夾的 ``local_file``。

槽位契約:**無輸入** → ``sources`` + ``meta``(ingestion 以
``pipeline.run({})`` 啟動,元件必須能無輸入執行;設定值走 ``__init__``
+ ``init_params``)。

- ``sources``:每筆是本地路徑(str / Path)或 ``ByteStream``(API 直接
  回內容時 —— 不必先落地成暫存檔)。
- ``meta``:與 sources 等長、一一對應;**每筆必須帶 ``doc_id``**(切片
  身分 ``chunk_id = "{doc_id}::chunk_{seq}"`` 的根;缺了會在下游
  ChunkMetaStamper 報錯)。doc_id 要跨執行穩定(upsert 與評估集都依賴)。

注意:sources 不是本地路徑時,``indexing.params.incremental: true`` 的
**檔案層**增量無從比對檔案雜湊,會退化為每次全量重 parse(切片層增量
仍會跳過內容未變的 embedding)。

掛載方式(configs/custom_ingestion_demo.yaml):

    import:
      method: custom
      params:
        file: examples/custom_modules/company_importer.py
        class: CompanyApiImporter
        content_type: text        # 供與 parsing 的相容性檢查;省略 = 跳過
        # init_params:
        #   base_url: https://km.example.com/api
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from haystack import component
from haystack.dataclasses import ByteStream

# logger 命名在 "rag.*" 底下:file: 與 class_path: 兩種載入方式都吃得到
# 框架的 log 檔設定(見 README「自訂方法」的「log 要看得到」)。
logger = logging.getLogger("rag.custom.company_importer")

# 示範用的假 API 回應(TODO(替換點):接上真正的來源後刪除)。
_FAKE_API_DOCUMENTS: list[dict[str, str]] = [
    {
        "dockey": "HR-0001",
        "contentTitle": "請假規定",
        "contentBody": "特休假依年資核給,請假需於三日前於差勤系統提出申請。"
        "婚假八日、喪假依親等三至八日。",
    },
    {
        "dockey": "HR-0002",
        "contentTitle": "薪資發放",
        "contentBody": "薪資於每月五日發放,遇假日提前至前一個工作日。",
    },
    {
        "dockey": "IT-0001",
        "contentTitle": "VPN 使用",
        "contentBody": "遠端工作需先安裝公司 VPN,帳號密碼與網域登入相同。"
        "連線異常請重啟客戶端後聯絡資訊服務台。",
    },
]


@component
class CompanyApiImporter:
    """從公司 API 拉文件,輸出 ByteStream sources 與帶 doc_id 的 meta。

    Args:
        base_url: 公司文件 API 端點;``None``(預設)回傳離線示範文件,
            骨架因此不接任何服務也能跑完。
        headers: 額外 HTTP 標頭(認證放這裡;建議 ``${ENV_VAR}`` 注入)。
        timeout: 單次請求逾時秒數。
    """

    def __init__(
        self,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url
        self.headers = dict(headers or {})
        self.timeout = timeout

    # sources 的型別註記:單一型別(list[str] / list[ByteStream])或**完整**
    # union(list[str | Path | ByteStream])都可;部分 union(如
    # list[str | ByteStream])在 Haystack 的型別比對下不相容,會被建構期擋下。
    @component.output_types(
        sources=list[str | Path | ByteStream], meta=list[dict[str, Any]]
    )
    def run(self) -> dict[str, Any]:
        """列出全部文件,回傳 ``{"sources": [...], "meta": [...]}``。

        TODO(替換點):換成公司 API 的真正呼叫,例如::

            import httpx

            response = httpx.get(
                f"{self.base_url}/documents", headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            rows = response.json()["returnData"]

        之後的邊界映射照舊:每筆文件一個 ByteStream(bytes 內容)+
        一筆 meta(``doc_id`` 用公司的穩定主鍵,其餘欄位無損保留,
        會一路跟著切片進索引)。
        """
        rows = _FAKE_API_DOCUMENTS if self.base_url is None else self._fetch_rows()
        sources: list[str | ByteStream] = []
        meta: list[dict[str, Any]] = []
        for row in rows:
            sources.append(
                ByteStream(data=row["contentBody"].encode("utf-8"))
            )
            meta.append(
                {
                    "doc_id": row["dockey"],  # 必填:切片身分的根,須跨執行穩定
                    "contentTitle": row["contentTitle"],
                    "importer": "custom",
                }
            )
        logger.debug("CompanyApiImporter:載入 %d 份文件", len(sources))
        return {"sources": sources, "meta": meta}

    def _fetch_rows(self) -> list[dict[str, str]]:
        raise NotImplementedError(
            f"CompanyApiImporter 尚未接上真正的文件 API(base_url={self.base_url});"
            "請在 _fetch_rows 換入公司 API 的呼叫,"
            "或移除 init_params.base_url 走離線示範文件"
        )

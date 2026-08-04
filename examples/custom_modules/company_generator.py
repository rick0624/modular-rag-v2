"""公司 LLM generation custom module 骨架(佔位範例,邏輯待替換)。

``generation`` 槽位的 custom module 範本:接公司自家的推論服務。內部
LLM 常常不是 OpenAI 相容格式(欄位名不同、要帶專案代號、回應包了一層
信封),內建的 ``gateway_openai_compatible`` 兜不過來時就寫這一支。

槽位契約:``messages: list[ChatMessage] → replies: list[ChatMessage]``
—— 也就是 Haystack 的 ChatGenerator 形狀。**prompt 仍由框架組**
(``prompt_template`` / ``system_prompt`` 照常寫在 YAML,ChatPromptBuilder
把切片與問題渲染好才送進來),這支元件只負責「把 messages 送出去、
把回答包回 ChatMessage」。

同一個類別也可以掛在其他 LLM 方法的 ``params.generator`` 底下
(``llm_rewrite`` / ``llm_decompose`` / reranking 的 ``llm`` 與
``llm_fact_check``),整條 pipeline 因此只走公司的推論服務。

掛載方式(configs/custom_demo.yaml):

    generation:
      method: custom
      params:
        file: examples/custom_modules/company_generator.py
        class: CompanyLLMGenerator
        init_params:
          endpoint: https://llm.example.com/v1/completions   # 不設 = 離線示範
        # prompt_template / system_prompt 與內建方法同款(選填)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from haystack import component
from haystack.dataclasses import ChatMessage

# logger 命名在 "rag.*" 底下:file: 與 class_path: 兩種載入方式都吃得到
# 框架的 log 檔設定(見 README「自訂方法」的「log 要看得到」)。
logger = logging.getLogger("rag.custom.company_generator")


@component
class CompanyLLMGenerator:
    """呼叫公司推論服務並回傳單一回覆。

    Args:
        endpoint: 推論服務的完整 URL;``None``(預設)走離線示範回覆,
            骨架因此不接任何服務也能跑完。
        model: 模型名稱,原樣放進請求;``None`` 時請求不帶該欄位。
        timeout: 單次請求的逾時秒數。
        headers: 額外 HTTP 標頭(認證放這裡;建議用 ``${ENV_VAR}`` 由
            環境變數注入,不要寫死在 YAML)。
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.headers = dict(headers or {})

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """把組好的 ``messages`` 送去生成,回傳 ``{"replies": [ChatMessage]}``。

        契約要點:回傳的清單**至少一則**(框架取 ``replies[0]``),且必須
        是 ``ChatMessage.from_assistant(...)``。``meta`` 會原樣進 ``query()``
        的 ``reply_meta`` 與 log,把 model / finish_reason / usage 放進去,
        事後才查得到「這個答案是誰生的、有沒有被截斷」。
        """
        started = time.perf_counter()
        if self.endpoint is None:
            answer, extra_meta = self._offline_reply(messages), {"offline": True}
        else:
            answer, extra_meta = self._call_company_llm(messages)
        meta: dict[str, Any] = {
            "model": self.model or "company-llm",
            "finish_reason": "stop",
            "latency_ms": (time.perf_counter() - started) * 1000,
            **extra_meta,
        }
        logger.debug("CompanyLLMGenerator:%d 則訊息 → %d 字", len(messages), len(answer))
        return {"replies": [ChatMessage.from_assistant(answer, meta=meta)]}

    def _call_company_llm(
        self, messages: list[ChatMessage]
    ) -> tuple[str, dict[str, Any]]:
        """TODO(替換點):換成公司推論服務的真正呼叫。

        典型長相(欄位名依公司 API 調整)::

            import httpx

            body = {
                "prompt": [
                    {"role": m.role.value, "text": m.text or ""} for m in messages
                ],
            }
            if self.model is not None:
                body["modelName"] = self.model
            response = httpx.post(
                self.endpoint, json=body, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["returnData"]["answer"], {"usage": data.get("usage")}

        失敗處理建議沿用框架的錯誤型別(``rag.errors.APICallError`` /
        ``APIResponseFormatError``):訊息要帶上端點與回應前幾百字,
        現場才查得下去。
        """
        raise NotImplementedError(
            f"CompanyLLMGenerator 尚未接上真正的推論服務(endpoint={self.endpoint});"
            "請在 _call_company_llm 換入公司 API 的呼叫,"
            "或移除 init_params.endpoint 走離線示範回覆"
        )

    @staticmethod
    def _offline_reply(messages: list[ChatMessage]) -> str:
        """離線示範回覆:引用 prompt 開頭,肉眼即可確認 prompt 組對了。"""
        last_user = next(
            (m.text or "" for m in reversed(messages) if m.role.value == "user"), ""
        )
        preview = " ".join(last_user.split())[:60]
        return f"[公司 LLM 骨架回答] 尚未接上推論服務(prompt 開頭:{preview}…)"

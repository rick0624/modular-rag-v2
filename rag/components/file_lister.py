"""Import 槽位:列出本地資料夾中的檔案,並產生穩定的 doc_id。

Haystack 的 converter 會把 ``meta`` 清單逐一合併進產出的 Document,
因此 doc_id 在這裡(pipeline 的最上游)就決定,不受 converter 的
``store_full_path`` 行為影響。doc_id = 檔案相對 ``input_dir`` 的
POSIX 路徑,同一來源每次執行都得到相同的 id(service/upsert 語意
與評估資料集都依賴這一點)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from haystack import component

from rag.errors import ComponentError


@component
class FileLister:
    """列出資料夾中的檔案,輸出 converter 需要的 ``sources`` 與 ``meta``。"""

    def __init__(
        self,
        input_dir: str,
        extensions: list[str],
        method_name: str = "local_file",
        recursive: bool = True,
    ) -> None:
        self.input_dir = input_dir
        self.extensions = {ext.lower() for ext in extensions}
        self.method_name = method_name
        self.recursive = recursive

    @component.output_types(sources=list[str], meta=list[dict[str, Any]])
    def run(self) -> dict[str, Any]:
        """掃描資料夾並回傳排序穩定的檔案清單。

        Raises:
            ComponentError: ``input_dir`` 不存在或不是資料夾。
        """
        base = Path(self.input_dir)
        if not base.is_dir():
            raise ComponentError(f"input_dir 不存在或不是資料夾:{base}")
        iterator = base.rglob("*") if self.recursive else base.glob("*")
        files = sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in self.extensions
        )
        meta = [
            {
                "doc_id": path.relative_to(base).as_posix(),
                "source": str(path),
                "importer": self.method_name,
            }
            for path in files
        ]
        return {"sources": [str(path) for path in files], "meta": meta}

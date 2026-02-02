from __future__ import annotations

from typing import Any
from unittest.mock import Mock
from pathlib import Path
from types import ModuleType
import sys


def _ensure_backend_data_stub() -> None:
    if "backend.data" in sys.modules:
        return
    backend_root = Path(__file__).resolve().parents[2]
    data_path = backend_root / "backend" / "data"
    pkg = ModuleType("backend.data")
    pkg.__path__ = [str(data_path)]
    sys.modules["backend.data"] = pkg


def _make_mock_docs_service() -> Mock:
    service = Mock()
    service.documents.return_value.batchUpdate.return_value.execute.return_value = {}
    return service


def format_docs_text(args: dict[str, Any]) -> dict[str, Any]:
    _ensure_backend_data_stub()
    from backend.blocks.google.docs import GoogleDocsFormatTextBlock

    block = GoogleDocsFormatTextBlock()
    service = _make_mock_docs_service()

    result = block._format_text(
        service,
        document_id=str(args.get("document_id", "")),
        start_index=int(args.get("start_index", 0)),
        end_index=int(args.get("end_index", 0)),
        bold=bool(args.get("bold", False)),
        italic=bool(args.get("italic", False)),
        underline=bool(args.get("underline", False)),
        font_size=int(args.get("font_size", 0) or 0),
        foreground_color=str(args.get("foreground_color", "")),
    )

    rgb = None
    if service.documents.return_value.batchUpdate.call_args:
        _, kwargs = service.documents.return_value.batchUpdate.call_args
        requests = kwargs["body"]["requests"]
        text_style = requests[0]["updateTextStyle"]["textStyle"]
        color = text_style.get("foregroundColor", {}).get("color", {}).get("rgbColor")
        if color:
            rgb = color

    return {"result": result, "rgb": rgb}


TOOLS = {
    "format_docs_text": format_docs_text,
}

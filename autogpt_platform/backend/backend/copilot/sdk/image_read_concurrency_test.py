import io
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from backend.copilot.model import ChatSession
from backend.copilot.sdk import e2b_file_tools
from backend.copilot.tool_image import ToolImage, prepare_image
from backend.copilot.tools import workspace_files


@pytest.fixture
def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "blue").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def preparation_thread_ids(monkeypatch):
    thread_ids = []

    def record_preparation(raw: bytes) -> ToolImage:
        thread_ids.append(threading.get_ident())
        return prepare_image(raw)

    monkeypatch.setattr(e2b_file_tools, "prepare_image", record_preparation)
    monkeypatch.setattr(workspace_files, "prepare_image", record_preparation)
    return thread_ids


async def test_workspace_image_preparation_runs_outside_the_event_loop(
    monkeypatch, png_bytes, preparation_thread_ids
):
    event_loop_thread = threading.get_ident()
    info = SimpleNamespace(
        id="image-file",
        name="page.png",
        path="/page.png",
        mime_type="image/png",
        size_bytes=len(png_bytes),
        metadata={},
    )
    manager = SimpleNamespace(
        get_file_info=AsyncMock(return_value=info),
        read_file_by_id=AsyncMock(return_value=png_bytes),
    )
    monkeypatch.setattr(
        workspace_files, "get_workspace_manager", AsyncMock(return_value=manager)
    )
    session = ChatSession.new("image-test-user", dry_run=False)
    result = await workspace_files.ReadWorkspaceFileTool()._execute(
        session.user_id, session, file_id="image-file"
    )
    assert result.image is not None
    assert len(preparation_thread_ids) == 1
    assert preparation_thread_ids[0] != event_loop_thread


@pytest.mark.parametrize("reader", ["local", "local_fallback", "sandbox"])
async def test_sdk_image_preparation_runs_outside_the_event_loop(
    monkeypatch, tmp_path, png_bytes, preparation_thread_ids, reader
):
    event_loop_thread = threading.get_ident()
    path = tmp_path / "page.png"
    path.write_bytes(png_bytes)
    monkeypatch.setattr(e2b_file_tools, "get_sdk_cwd", lambda: str(tmp_path))
    stream = MagicMock()
    stream.__aiter__.return_value = [png_bytes]
    sandbox = (
        SimpleNamespace(
            files=SimpleNamespace(
                get_info=AsyncMock(return_value=SimpleNamespace(size=len(png_bytes))),
                read=AsyncMock(return_value=stream),
            )
        )
        if reader == "sandbox"
        else None
    )
    monkeypatch.setattr(e2b_file_tools, "_get_sandbox", lambda: sandbox)
    monkeypatch.setattr(e2b_file_tools, "is_sdk_tool_path", lambda _: False)
    monkeypatch.setattr(e2b_file_tools, "resolve_sandbox_path", lambda path: path)
    if reader == "local_fallback":
        monkeypatch.setattr(e2b_file_tools, "_is_allowed_local", lambda _: False)
    result = await e2b_file_tools._handle_read_file({"file_path": str(path)})
    assert result["content"][-1]["type"] == "image"
    assert len(preparation_thread_ids) == 1
    assert preparation_thread_ids[0] != event_loop_thread

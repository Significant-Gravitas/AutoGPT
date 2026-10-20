import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pypdfium2
import pytest
from PIL import Image

from backend.copilot.baseline import service as baseline
from backend.copilot.google_data_redaction import GOOGLE_DATA_METADATA_KEY
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.sdk import e2b_file_tools, tool_adapter
from backend.copilot.tool_image import MAX_IMAGE_BYTES
from backend.copilot.tools import workspace_files
from backend.copilot.tools.base import BaseTool
from backend.copilot.transcript_builder import TranscriptBuilder
from backend.util.tool_call_loop import LLMLoopResponse, LLMToolCall


@pytest.fixture
def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.effect_noise((512, 512), 100).convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def workspace(monkeypatch, png_bytes):
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
        get_file_info_by_path=AsyncMock(return_value=info),
        read_file_by_id=AsyncMock(return_value=png_bytes),
    )
    monkeypatch.setattr(
        workspace_files, "get_workspace_manager", AsyncMock(return_value=manager)
    )
    monkeypatch.setattr(BaseTool, "_gate", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(BaseTool, "_released_read", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "backend.copilot.gate.reads.screen_read", AsyncMock(return_value=None)
    )
    monkeypatch.setattr("backend.copilot.tools.base._record_activity", AsyncMock())
    monkeypatch.setattr(
        "backend.copilot.tools.track_tool_called", lambda **kwargs: None
    )
    return manager


@pytest.mark.parametrize("args", [{"file_id": "image-file"}, {"path": "/page.png"}])
async def test_workspace_image_reaches_sdk_after_truncation(workspace, png_bytes, args):
    session = ChatSession.new(user_id="image-test-user", dry_run=False)
    tool_adapter.set_execution_context(user_id="image-test-user", session=session)
    handler = tool_adapter._make_truncating_wrapper(
        tool_adapter.create_tool_handler(workspace_files.ReadWorkspaceFileTool()),
        "read_workspace_file",
    )
    result = await handler(args)
    images = [block for block in result["content"] if block["type"] == "image"]
    assert len(images) == 1
    assert base64.b64decode(images[0]["data"], validate=True) == png_bytes


@pytest.mark.parametrize("relative", [False, True])
async def test_working_directory_image_reaches_sdk(
    tmp_path, monkeypatch, png_bytes, relative
):
    path = tmp_path / "page.png"
    path.write_bytes(png_bytes)
    monkeypatch.setattr(e2b_file_tools, "get_sdk_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(e2b_file_tools, "_get_sandbox", lambda: None)
    result = await e2b_file_tools._handle_read_file(
        {"file_path": path.name if relative else str(path)}
    )
    images = [block for block in result["content"] if block["type"] == "image"]
    assert len(images) == 1
    assert base64.b64decode(images[0]["data"], validate=True) == png_bytes


async def test_workspace_image_reaches_baseline_request_and_transcript(
    workspace, png_bytes
):
    session = ChatSession.new("image-test-user", dry_run=False)
    state = baseline._BaselineStreamState()
    call = LLMToolCall(
        id="call-image",
        name="read_workspace_file",
        arguments='{"file_id":"image-file"}',
    )
    result = await baseline._baseline_tool_executor(
        call,
        [],
        state=state,
        user_id=session.user_id,
        session=session,
        disabled_groups=[],
        disabled_tools=frozenset(),
    )
    messages = []
    response = LLMLoopResponse(response_text=None, tool_calls=[call], raw_response=None)
    baseline._mutate_openai_messages(messages, response, [result])
    assert [message["role"] for message in messages] == ["assistant", "tool", "user"]
    url = messages[-1]["content"][1]["image_url"]["url"]
    assert base64.b64decode(url.split(",", 1)[1], validate=True) == png_bytes
    outputs = [
        event
        for event in state.emitted_events
        if isinstance(event, StreamToolOutputAvailable)
    ]
    assert len(outputs) == 1
    assert len(outputs[0].to_sse()) < 2000
    assert base64.b64encode(png_bytes).decode() not in outputs[0].model_dump_json()
    assert "content_base64" not in result.content
    builder = TranscriptBuilder()
    baseline._record_turn_to_transcript(
        response, [result], transcript_builder=builder, model="test-model"
    )
    assert result.image_blocks[0]["source"]["data"] in builder.to_jsonl()


async def test_sandbox_image_reaches_sdk_wrapper(monkeypatch, png_bytes):
    stream = MagicMock()
    stream.__aiter__.return_value = [png_bytes]
    sandbox = SimpleNamespace(
        files=SimpleNamespace(
            get_info=AsyncMock(return_value=SimpleNamespace(size=len(png_bytes))),
            read=AsyncMock(return_value=stream),
        )
    )
    monkeypatch.setattr(e2b_file_tools, "_get_sandbox", lambda: sandbox)
    monkeypatch.setattr(e2b_file_tools, "is_sdk_tool_path", lambda path: False)
    monkeypatch.setattr(e2b_file_tools, "resolve_sandbox_path", lambda path: path)
    tool_adapter.set_execution_context(user_id=None, session=None)
    handler = tool_adapter._make_truncating_wrapper(
        e2b_file_tools._handle_read_file, "read_file"
    )
    result = await handler({"file_path": "/home/user/frame.png"})
    assert result["content"][-1]["type"] == "image"
    assert base64.b64decode(result["content"][-1]["data"], validate=True) == png_bytes
    sandbox.files.read.assert_awaited_once_with("/home/user/frame.png", format="stream")
    stream.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("filename", ["scan.pdf", "movie.mp4", "archive.zip"])
async def test_sandbox_binary_rejected_without_downloading(monkeypatch, filename):
    sandbox = SimpleNamespace(
        files=SimpleNamespace(get_info=AsyncMock(), read=AsyncMock(return_value=b""))
    )
    monkeypatch.setattr(e2b_file_tools, "_get_sandbox", lambda: sandbox)
    monkeypatch.setattr(e2b_file_tools, "is_sdk_tool_path", lambda _: False)
    monkeypatch.setattr(e2b_file_tools, "resolve_sandbox_path", lambda path: path)
    result = await e2b_file_tools._handle_read_file(
        {"file_path": f"/home/user/{filename}"}
    )
    assert result["isError"]
    assert "has not been viewed" in result["content"][0]["text"]
    sandbox.files.read.assert_not_awaited()
    sandbox.files.get_info.assert_not_awaited()


@pytest.mark.parametrize(
    "args", [{"force_download_url": True}, {"offset": 1}, {"length": 10}]
)
async def test_metadata_and_text_ranges_do_not_return_pixels(workspace, args):
    session = ChatSession.new("image-test-user", dry_run=False)
    result = await workspace_files.ReadWorkspaceFileTool()._execute(
        session.user_id, session, file_id="image-file", **args
    )
    assert result.image is None
    workspace.read_file_by_id.assert_not_awaited()


async def test_workspace_oversize_metadata_prevents_read(workspace):
    workspace.get_file_info.return_value.size_bytes = MAX_IMAGE_BYTES + 1
    session = ChatSession.new("image-test-user", dry_run=False)
    result = await workspace_files.ReadWorkspaceFileTool()._execute(
        session.user_id, session, file_id="image-file"
    )
    assert result.image is None
    assert "exceeds 3 MiB" in result.message
    workspace.read_file_by_id.assert_not_awaited()


async def test_workspace_actual_bytes_are_checked(workspace):
    workspace.read_file_by_id.return_value = b"x" * (MAX_IMAGE_BYTES + 1)
    session = ChatSession.new("image-test-user", dry_run=False)
    result = await workspace_files.ReadWorkspaceFileTool()._execute(
        session.user_id, session, file_id="image-file"
    )
    assert result.image is None
    assert "exceeds 3 MiB" in result.message


async def test_workspace_image_preserves_google_data_marking(workspace):
    workspace.get_file_info.return_value.metadata[GOOGLE_DATA_METADATA_KEY] = True
    session = ChatSession.new("image-test-user", dry_run=False)
    result = await workspace_files.ReadWorkspaceFileTool()._execute(
        session.user_id, session, file_id="image-file"
    )
    assert result.image is not None
    assert result.holds_google_data


async def test_workspace_denial_never_reads_pixels(workspace):
    workspace.get_file_info.return_value = None
    session = ChatSession.new("image-test-user", dry_run=False)
    result = await workspace_files.ReadWorkspaceFileTool()._execute(
        session.user_id, session, file_id="image-file"
    )
    assert result.image is None
    workspace.read_file_by_id.assert_not_awaited()


async def test_local_path_escape_does_not_return_pixels(
    tmp_path, monkeypatch, png_bytes
):
    working = tmp_path / "working"
    working.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(png_bytes)
    monkeypatch.setattr(e2b_file_tools, "get_sdk_cwd", lambda: str(working))
    monkeypatch.setattr(e2b_file_tools, "_get_sandbox", lambda: None)
    result = await e2b_file_tools._handle_read_file({"file_path": str(outside)})
    assert result["isError"]
    assert all(block["type"] != "image" for block in result["content"])


@pytest.mark.parametrize("args", [{"offset": 1}, {"limit": 10}, {"limit": 2000}])
async def test_local_image_rejects_text_ranges(tmp_path, monkeypatch, png_bytes, args):
    path = tmp_path / "page.png"
    path.write_bytes(png_bytes)
    monkeypatch.setattr(e2b_file_tools, "get_sdk_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(e2b_file_tools, "_get_sandbox", lambda: None)
    result = await e2b_file_tools._handle_read_file({"file_path": str(path), **args})
    assert result["isError"]
    assert "Read the whole image" in result["content"][0]["text"]
    assert all(block["type"] != "image" for block in result["content"])


async def test_pdf_gives_render_guidance(workspace):
    workspace.get_file_info.return_value.mime_type = "application/pdf"
    workspace.get_file_info.return_value.name = "scan.pdf"
    session = ChatSession.new("image-test-user", dry_run=False)
    result = await workspace_files.ReadWorkspaceFileTool()._execute(
        session.user_id, session, file_id="image-file"
    )
    assert result.image is None
    assert "Render PDF pages" in result.message
    workspace.read_file_by_id.assert_not_awaited()


async def test_rendered_scanned_pdf_page_is_viewable(tmp_path, monkeypatch):
    pdf_data = io.BytesIO()
    Image.new("RGB", (300, 400), "red").save(pdf_data, format="PDF")
    path = tmp_path / "rendered-page.png"
    with pypdfium2.PdfDocument(pdf_data.getvalue()) as document:
        page = document[0]
        bitmap = page.render(scale=1)
        bitmap.to_pil().save(path, format="PNG")
        bitmap.close()
        page.close()
    monkeypatch.setattr(e2b_file_tools, "get_sdk_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(e2b_file_tools, "_get_sandbox", lambda: None)
    result = await e2b_file_tools._handle_read_file({"file_path": str(path)})
    assert (
        base64.b64decode(result["content"][-1]["data"], validate=True)
        == path.read_bytes()
    )


async def test_workspace_image_mime_is_detected_from_bytes(workspace):
    workspace.get_file_info.return_value.mime_type = "application/octet-stream"
    session = ChatSession.new("image-test-user", dry_run=False)
    result = await workspace_files.ReadWorkspaceFileTool()._execute(
        session.user_id, session, file_id="image-file"
    )
    assert result.image is not None
    assert result.image.mime_type == "image/png"


async def test_svg_keeps_its_download_handle_and_conversion_guidance(workspace):
    workspace.get_file_info.return_value.mime_type = "image/svg+xml"
    workspace.get_file_info.return_value.name = "chart.svg"
    session = ChatSession.new("image-test-user", dry_run=False)
    result = await workspace_files.ReadWorkspaceFileTool()._execute(
        session.user_id, session, file_id="image-file"
    )
    assert result.image is None
    assert result.download_url == "workspace://image-file"
    assert "convert" in result.message.lower()
    workspace.read_file_by_id.assert_not_awaited()


async def test_invalid_saved_image_acknowledges_local_copy(workspace, monkeypatch):
    workspace.read_file_by_id.return_value = b"invalid"
    monkeypatch.setattr(
        workspace_files, "save_to_workdir", AsyncMock(return_value="page.png")
    )
    session = ChatSession.new("image-test-user", dry_run=False)
    result = await workspace_files.ReadWorkspaceFileTool()._execute(
        session.user_id, session, file_id="image-file", save_to_path="page.png"
    )
    assert result.image is None
    assert "saved to page.png" in result.message

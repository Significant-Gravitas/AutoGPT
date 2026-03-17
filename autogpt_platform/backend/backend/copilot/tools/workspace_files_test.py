"""Tests for workspace file tool helpers and path validation."""

import base64
import os
import shutil
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.context import SDK_PROJECTS_DIR, _current_project_dir
from backend.copilot.tools._test_data import make_session, setup_test_data
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.workspace_files import (
    _MAX_LOCAL_TOOL_RESULT_BYTES,
    DeleteWorkspaceFileTool,
    ListWorkspaceFilesTool,
    ReadWorkspaceFileTool,
    WorkspaceDeleteResponse,
    WorkspaceFileContentResponse,
    WorkspaceFileListResponse,
    WorkspaceWriteResponse,
    WriteWorkspaceFileTool,
    _read_local_tool_result,
    _resolve_write_content,
    _validate_ephemeral_path,
)

# Re-export so pytest discovers the session-scoped fixture
setup_test_data = setup_test_data

# We need to mock make_session_path to return a known temp dir for tests.
# The real one uses WORKSPACE_PREFIX = "/tmp/copilot-"


@pytest.fixture
def ephemeral_dir(tmp_path, monkeypatch):
    """Create a temp dir that acts as the ephemeral session directory."""
    session_dir = tmp_path / "copilot-test-session"
    session_dir.mkdir()

    monkeypatch.setattr(
        "backend.copilot.tools.workspace_files.make_session_path",
        lambda session_id: str(session_dir),
    )
    return session_dir


# ---------------------------------------------------------------------------
# _validate_ephemeral_path
# ---------------------------------------------------------------------------


class TestValidateEphemeralPath:
    def test_valid_path(self, ephemeral_dir):
        target = ephemeral_dir / "file.txt"
        target.touch()
        result = _validate_ephemeral_path(
            str(target), param_name="test", session_id="s1"
        )
        assert isinstance(result, str)
        assert result == os.path.realpath(str(target))

    def test_path_traversal_rejected(self, ephemeral_dir):
        evil_path = str(ephemeral_dir / ".." / "etc" / "passwd")
        result = _validate_ephemeral_path(evil_path, param_name="test", session_id="s1")
        # Should return ErrorResponse
        from backend.copilot.tools.models import ErrorResponse

        assert isinstance(result, ErrorResponse)

    def test_different_session_rejected(self, ephemeral_dir, tmp_path):
        other_dir = tmp_path / "copilot-evil-session"
        other_dir.mkdir()
        target = other_dir / "steal.txt"
        target.touch()
        result = _validate_ephemeral_path(
            str(target), param_name="test", session_id="s1"
        )
        from backend.copilot.tools.models import ErrorResponse

        assert isinstance(result, ErrorResponse)

    def test_symlink_escape_rejected(self, ephemeral_dir, tmp_path):
        """Symlink inside session dir pointing outside should be rejected."""
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret")
        symlink = ephemeral_dir / "link.txt"
        symlink.symlink_to(outside_file)
        result = _validate_ephemeral_path(
            str(symlink), param_name="test", session_id="s1"
        )
        from backend.copilot.tools.models import ErrorResponse

        assert isinstance(result, ErrorResponse)

    def test_nested_path_valid(self, ephemeral_dir):
        nested = ephemeral_dir / "subdir" / "deep"
        nested.mkdir(parents=True)
        target = nested / "data.csv"
        target.touch()
        result = _validate_ephemeral_path(
            str(target), param_name="test", session_id="s1"
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _resolve_write_content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
class TestResolveWriteContent:
    async def test_no_sources_returns_error(self):
        from backend.copilot.tools.models import ErrorResponse

        result = await _resolve_write_content(None, None, None, "s1")
        assert isinstance(result, ErrorResponse)

    async def test_multiple_sources_returns_error(self):
        from backend.copilot.tools.models import ErrorResponse

        result = await _resolve_write_content("text", "b64data", None, "s1")
        assert isinstance(result, ErrorResponse)

    async def test_plain_text_content(self):
        result = await _resolve_write_content("hello world", None, None, "s1")
        assert result == b"hello world"

    async def test_base64_content(self):
        raw = b"binary data"
        b64 = base64.b64encode(raw).decode()
        result = await _resolve_write_content(None, b64, None, "s1")
        assert result == raw

    async def test_invalid_base64_returns_error(self):
        from backend.copilot.tools.models import ErrorResponse

        result = await _resolve_write_content(None, "not-valid-b64!!!", None, "s1")
        assert isinstance(result, ErrorResponse)
        assert "base64" in result.message.lower()

    async def test_source_path(self, ephemeral_dir):
        target = ephemeral_dir / "input.txt"
        target.write_bytes(b"file content")
        result = await _resolve_write_content(None, None, str(target), "s1")
        assert result == b"file content"

    async def test_source_path_not_found(self, ephemeral_dir):
        from backend.copilot.tools.models import ErrorResponse

        missing = str(ephemeral_dir / "nope.txt")
        result = await _resolve_write_content(None, None, missing, "s1")
        assert isinstance(result, ErrorResponse)

    async def test_source_path_outside_ephemeral(self, ephemeral_dir, tmp_path):
        from backend.copilot.tools.models import ErrorResponse

        outside = tmp_path / "outside.txt"
        outside.write_text("nope")
        result = await _resolve_write_content(None, None, str(outside), "s1")
        assert isinstance(result, ErrorResponse)

    async def test_empty_string_sources_treated_as_none(self):
        from backend.copilot.tools.models import ErrorResponse

        # All empty strings → same as no sources
        result = await _resolve_write_content("", "", "", "s1")
        assert isinstance(result, ErrorResponse)

    async def test_empty_string_source_path_with_text(self):
        # source_path="" should be normalised to None, so only content counts
        result = await _resolve_write_content("hello", "", "", "s1")
        assert result == b"hello"


# ---------------------------------------------------------------------------
# E2E: workspace file tool round-trip (write → list → read → delete)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_workspace_file_round_trip(setup_test_data):
    """E2E: write a file, list it, read it back (with save_to_path), then delete it."""
    user = setup_test_data["user"]
    session = make_session(user.id)
    session_id = session.session_id

    # ---- Write ----
    write_tool = WriteWorkspaceFileTool()
    write_resp = await write_tool._execute(
        user_id=user.id,
        session=session,
        filename="test_round_trip.txt",
        content="Hello from e2e test!",
    )
    assert isinstance(write_resp, WorkspaceWriteResponse), write_resp.message
    file_id = write_resp.file_id

    # ---- List ----
    list_tool = ListWorkspaceFilesTool()
    list_resp = await list_tool._execute(user_id=user.id, session=session)
    assert isinstance(list_resp, WorkspaceFileListResponse), list_resp.message
    assert any(f.file_id == file_id for f in list_resp.files)

    # ---- Read (inline) ----
    read_tool = ReadWorkspaceFileTool()
    read_resp = await read_tool._execute(
        user_id=user.id, session=session, file_id=file_id
    )
    from backend.copilot.tools.workspace_files import WorkspaceFileContentResponse

    assert isinstance(read_resp, WorkspaceFileContentResponse), read_resp.message
    decoded = base64.b64decode(read_resp.content_base64).decode()
    assert decoded == "Hello from e2e test!"

    # ---- Read with save_to_path ----
    from backend.copilot.tools.sandbox import make_session_path

    ephemeral_dir = make_session_path(session_id)
    os.makedirs(ephemeral_dir, exist_ok=True)
    save_path = os.path.join(ephemeral_dir, "saved_copy.txt")

    read_resp2 = await read_tool._execute(
        user_id=user.id, session=session, file_id=file_id, save_to_path=save_path
    )
    assert not isinstance(read_resp2, type(None))
    assert os.path.exists(save_path)
    with open(save_path) as f:
        assert f.read() == "Hello from e2e test!"

    # ---- Delete ----
    delete_tool = DeleteWorkspaceFileTool()
    del_resp = await delete_tool._execute(
        user_id=user.id, session=session, file_id=file_id
    )
    assert isinstance(del_resp, WorkspaceDeleteResponse), del_resp.message
    assert del_resp.success is True

    # Verify file is gone
    list_resp2 = await list_tool._execute(user_id=user.id, session=session)
    assert isinstance(list_resp2, WorkspaceFileListResponse)
    assert not any(f.file_id == file_id for f in list_resp2.files)


# ---------------------------------------------------------------------------
# Ranged reads (offset / length)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_read_workspace_file_with_offset_and_length(setup_test_data):
    """Read a slice of a text file using offset and length."""
    user = setup_test_data["user"]
    session = make_session(user.id)

    # Write a known-content file
    content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 100  # 2600 chars
    write_tool = WriteWorkspaceFileTool()
    write_resp = await write_tool._execute(
        user_id=user.id,
        session=session,
        filename="ranged_test.txt",
        content=content,
    )
    assert isinstance(write_resp, WorkspaceWriteResponse), write_resp.message
    file_id = write_resp.file_id

    from backend.copilot.tools.workspace_files import WorkspaceFileContentResponse

    read_tool = ReadWorkspaceFileTool()

    # Read with offset=100, length=50
    resp = await read_tool._execute(
        user_id=user.id, session=session, file_id=file_id, offset=100, length=50
    )
    assert isinstance(resp, WorkspaceFileContentResponse), resp.message
    decoded = base64.b64decode(resp.content_base64).decode()
    assert decoded == content[100:150]
    assert "100" in resp.message
    assert "2,600" in resp.message  # total chars (comma-formatted)

    # Read with offset only (no length) — returns from offset to end
    resp2 = await read_tool._execute(
        user_id=user.id, session=session, file_id=file_id, offset=2500
    )
    assert isinstance(resp2, WorkspaceFileContentResponse)
    decoded2 = base64.b64decode(resp2.content_base64).decode()
    assert decoded2 == content[2500:]
    assert len(decoded2) == 100

    # Read with offset beyond file length — returns empty string
    resp3 = await read_tool._execute(
        user_id=user.id, session=session, file_id=file_id, offset=9999, length=10
    )
    assert isinstance(resp3, WorkspaceFileContentResponse)
    decoded3 = base64.b64decode(resp3.content_base64).decode()
    assert decoded3 == ""

    # Cleanup
    delete_tool = DeleteWorkspaceFileTool()
    await delete_tool._execute(user_id=user.id, session=session, file_id=file_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_write_workspace_file_source_path(setup_test_data):
    """E2E: write a file from ephemeral source_path to workspace."""
    user = setup_test_data["user"]
    session = make_session(user.id)
    session_id = session.session_id

    # Create a file in the ephemeral dir
    from backend.copilot.tools.sandbox import make_session_path

    ephemeral_dir = make_session_path(session_id)
    os.makedirs(ephemeral_dir, exist_ok=True)
    source = os.path.join(ephemeral_dir, "generated_output.csv")
    with open(source, "w") as f:
        f.write("col1,col2\n1,2\n")

    write_tool = WriteWorkspaceFileTool()
    write_resp = await write_tool._execute(
        user_id=user.id,
        session=session,
        filename="output.csv",
        source_path=source,
    )
    assert isinstance(write_resp, WorkspaceWriteResponse), write_resp.message

    # Clean up
    delete_tool = DeleteWorkspaceFileTool()
    await delete_tool._execute(
        user_id=user.id, session=session, file_id=write_resp.file_id
    )


# ---------------------------------------------------------------------------
# _read_local_tool_result — local disk fallback for SDK tool-result files
# ---------------------------------------------------------------------------

_CONV_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


class TestReadLocalToolResult:
    """Tests for _read_local_tool_result (local disk fallback)."""

    def _make_tool_result(self, encoded: str, filename: str, content: bytes) -> str:
        """Create a tool-results file and return its path."""
        tool_dir = os.path.join(SDK_PROJECTS_DIR, encoded, _CONV_UUID, "tool-results")
        os.makedirs(tool_dir, exist_ok=True)
        filepath = os.path.join(tool_dir, filename)
        with open(filepath, "wb") as f:
            f.write(content)
        return filepath

    def _cleanup(self, encoded: str) -> None:
        shutil.rmtree(os.path.join(SDK_PROJECTS_DIR, encoded), ignore_errors=True)

    def test_read_text_file(self):
        """Read a UTF-8 text tool-result file."""
        encoded = "-tmp-copilot-local-read-text"
        path = self._make_tool_result(encoded, "output.txt", b"hello world")
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local_tool_result(path, 0, None, "s1")
            assert isinstance(result, WorkspaceFileContentResponse)
            decoded = base64.b64decode(result.content_base64).decode("utf-8")
            assert decoded == "hello world"
            assert "text/plain" in result.mime_type
        finally:
            _current_project_dir.reset(token)
            self._cleanup(encoded)

    def test_read_text_with_offset(self):
        """Read a slice of a text file using char_offset and char_length."""
        encoded = "-tmp-copilot-local-read-offset"
        path = self._make_tool_result(encoded, "data.txt", b"ABCDEFGHIJ")
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local_tool_result(path, 3, 4, "s1")
            assert isinstance(result, WorkspaceFileContentResponse)
            decoded = base64.b64decode(result.content_base64).decode("utf-8")
            assert decoded == "DEFG"
        finally:
            _current_project_dir.reset(token)
            self._cleanup(encoded)

    def test_read_binary_file(self):
        """Binary files are returned as raw base64."""
        encoded = "-tmp-copilot-local-read-binary"
        binary_data = bytes(range(256))
        path = self._make_tool_result(encoded, "image.png", binary_data)
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local_tool_result(path, 0, None, "s1")
            assert isinstance(result, WorkspaceFileContentResponse)
            decoded = base64.b64decode(result.content_base64)
            assert decoded == binary_data
            assert "binary" in result.message
        finally:
            _current_project_dir.reset(token)
            self._cleanup(encoded)

    def test_disallowed_path_rejected(self):
        """Paths not under allowed directories are rejected."""
        result = _read_local_tool_result("/etc/passwd", 0, None, "s1")
        assert isinstance(result, ErrorResponse)
        assert "not allowed" in result.message.lower()

    def test_file_not_found(self):
        """Missing files return an error."""
        encoded = "-tmp-copilot-local-read-missing"
        tool_dir = os.path.join(SDK_PROJECTS_DIR, encoded, _CONV_UUID, "tool-results")
        os.makedirs(tool_dir, exist_ok=True)
        path = os.path.join(tool_dir, "nope.txt")
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local_tool_result(path, 0, None, "s1")
            assert isinstance(result, ErrorResponse)
            assert "not found" in result.message.lower()
        finally:
            _current_project_dir.reset(token)
            self._cleanup(encoded)

    def test_file_too_large(self, monkeypatch):
        """Files exceeding the size limit are rejected."""
        encoded = "-tmp-copilot-local-read-large"
        # Create a small file but fake os.path.getsize to return a huge value
        path = self._make_tool_result(encoded, "big.txt", b"small")
        token = _current_project_dir.set(encoded)
        monkeypatch.setattr(
            "os.path.getsize", lambda _: _MAX_LOCAL_TOOL_RESULT_BYTES + 1
        )
        try:
            result = _read_local_tool_result(path, 0, None, "s1")
            assert isinstance(result, ErrorResponse)
            assert "too large" in result.message.lower()
        finally:
            _current_project_dir.reset(token)
            self._cleanup(encoded)

    def test_offset_beyond_file_length(self):
        """Offset past end-of-file returns empty content."""
        encoded = "-tmp-copilot-local-read-past-eof"
        path = self._make_tool_result(encoded, "short.txt", b"abc")
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local_tool_result(path, 999, 10, "s1")
            assert isinstance(result, WorkspaceFileContentResponse)
            decoded = base64.b64decode(result.content_base64).decode("utf-8")
            assert decoded == ""
        finally:
            _current_project_dir.reset(token)
            self._cleanup(encoded)

    def test_zero_length_read(self):
        """Requesting zero characters returns empty content."""
        encoded = "-tmp-copilot-local-read-zero-len"
        path = self._make_tool_result(encoded, "data.txt", b"ABCDEF")
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local_tool_result(path, 2, 0, "s1")
            assert isinstance(result, WorkspaceFileContentResponse)
            decoded = base64.b64decode(result.content_base64).decode("utf-8")
            assert decoded == ""
        finally:
            _current_project_dir.reset(token)
            self._cleanup(encoded)

    def test_mime_type_from_json_extension(self):
        """JSON files get application/json MIME type, not hardcoded text/plain."""
        encoded = "-tmp-copilot-local-read-json"
        path = self._make_tool_result(encoded, "result.json", b'{"key": "value"}')
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local_tool_result(path, 0, None, "s1")
            assert isinstance(result, WorkspaceFileContentResponse)
            assert result.mime_type == "application/json"
        finally:
            _current_project_dir.reset(token)
            self._cleanup(encoded)

    def test_mime_type_from_png_extension(self):
        """Binary .png files get image/png MIME type via mimetypes."""
        encoded = "-tmp-copilot-local-read-png-mime"
        binary_data = bytes(range(256))
        path = self._make_tool_result(encoded, "chart.png", binary_data)
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local_tool_result(path, 0, None, "s1")
            assert isinstance(result, WorkspaceFileContentResponse)
            assert result.mime_type == "image/png"
        finally:
            _current_project_dir.reset(token)
            self._cleanup(encoded)

    def test_explicit_sdk_cwd_parameter(self):
        """The sdk_cwd parameter overrides get_sdk_cwd() for path validation."""
        encoded = "-tmp-copilot-local-read-sdkcwd"
        path = self._make_tool_result(encoded, "out.txt", b"content")
        token = _current_project_dir.set(encoded)
        try:
            # Pass sdk_cwd explicitly — should still succeed because the path
            # is under SDK_PROJECTS_DIR which is always allowed.
            result = _read_local_tool_result(
                path, 0, None, "s1", sdk_cwd="/tmp/copilot-test"
            )
            assert isinstance(result, WorkspaceFileContentResponse)
            decoded = base64.b64decode(result.content_base64).decode("utf-8")
            assert decoded == "content"
        finally:
            _current_project_dir.reset(token)
            self._cleanup(encoded)

    def test_offset_with_no_length_reads_to_end(self):
        """When char_length is None, read from offset to end of file."""
        encoded = "-tmp-copilot-local-read-offset-noLen"
        path = self._make_tool_result(encoded, "data.txt", b"0123456789")
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local_tool_result(path, 5, None, "s1")
            assert isinstance(result, WorkspaceFileContentResponse)
            decoded = base64.b64decode(result.content_base64).decode("utf-8")
            assert decoded == "56789"
        finally:
            _current_project_dir.reset(token)
            self._cleanup(encoded)


# ---------------------------------------------------------------------------
# ReadWorkspaceFileTool fallback to _read_local_tool_result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_read_workspace_file_falls_back_to_local_tool_result(setup_test_data):
    """When _resolve_file returns ErrorResponse for an allowed local path,
    ReadWorkspaceFileTool should fall back to _read_local_tool_result."""
    user = setup_test_data["user"]
    session = make_session(user.id)

    # Create a real tool-result file on disk so the fallback can read it.
    encoded = "-tmp-copilot-fallback-test"
    conv_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    tool_dir = os.path.join(SDK_PROJECTS_DIR, encoded, conv_uuid, "tool-results")
    os.makedirs(tool_dir, exist_ok=True)
    filepath = os.path.join(tool_dir, "result.txt")
    with open(filepath, "w") as f:
        f.write("fallback content")

    token = _current_project_dir.set(encoded)
    try:
        # Mock _resolve_file to return an ErrorResponse (simulating "file not
        # found in workspace") so the fallback branch is exercised.
        mock_resolve = AsyncMock(
            return_value=ErrorResponse(
                message="File not found at path: result.txt",
                session_id=session.session_id,
            )
        )
        with patch("backend.copilot.tools.workspace_files._resolve_file", mock_resolve):
            read_tool = ReadWorkspaceFileTool()
            result = await read_tool._execute(
                user_id=user.id,
                session=session,
                path=filepath,
            )

        # Should have fallen back to _read_local_tool_result and succeeded.
        assert isinstance(result, WorkspaceFileContentResponse), (
            f"Expected fallback to local read, got {type(result).__name__}: "
            f"{getattr(result, 'message', '')}"
        )
        decoded = base64.b64decode(result.content_base64).decode("utf-8")
        assert decoded == "fallback content"
        mock_resolve.assert_awaited_once()
    finally:
        _current_project_dir.reset(token)
        shutil.rmtree(os.path.join(SDK_PROJECTS_DIR, encoded), ignore_errors=True)


@pytest.mark.asyncio(loop_scope="session")
async def test_read_workspace_file_no_fallback_when_resolve_succeeds(setup_test_data):
    """When _resolve_file succeeds, the local-disk fallback must NOT be invoked."""
    user = setup_test_data["user"]
    session = make_session(user.id)

    fake_file_id = "fake-file-id-001"
    fake_content = b"workspace content"

    # Build a minimal file_info stub that the tool's happy-path needs.
    class _FakeFileInfo:
        id = fake_file_id
        name = "result.json"
        path = "/result.json"
        mime_type = "text/plain"
        size_bytes = len(fake_content)

    mock_resolve = AsyncMock(return_value=(fake_file_id, _FakeFileInfo()))

    mock_manager = AsyncMock()
    mock_manager.read_file_by_id = AsyncMock(return_value=fake_content)

    with (
        patch("backend.copilot.tools.workspace_files._resolve_file", mock_resolve),
        patch(
            "backend.copilot.tools.workspace_files.get_workspace_manager",
            AsyncMock(return_value=mock_manager),
        ),
        patch(
            "backend.copilot.tools.workspace_files._read_local_tool_result"
        ) as patched_local,
    ):
        read_tool = ReadWorkspaceFileTool()
        result = await read_tool._execute(
            user_id=user.id,
            session=session,
            file_id=fake_file_id,
        )

    # Fallback must not have been called.
    patched_local.assert_not_called()
    # Normal workspace path must have produced a content response.
    assert isinstance(result, WorkspaceFileContentResponse)
    assert base64.b64decode(result.content_base64) == fake_content

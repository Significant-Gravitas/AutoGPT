import io
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.workspace.routes import router
from backend.data.workspace import Workspace, WorkspaceFile

app = fastapi.FastAPI()
app.include_router(router)


@app.exception_handler(ValueError)
async def _value_error_handler(
    request: fastapi.Request, exc: ValueError
) -> fastapi.responses.JSONResponse:
    """Mirror the production ValueError → 400 mapping from the REST app."""
    return fastapi.responses.JSONResponse(status_code=400, content={"detail": str(exc)})


client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _make_workspace(user_id: str = "test-user-id") -> Workspace:
    return Workspace(
        id="ws-001",
        user_id=user_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_file(**overrides) -> WorkspaceFile:
    defaults = {
        "id": "file-001",
        "workspace_id": "ws-001",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "name": "test.txt",
        "path": "/test.txt",
        "storage_path": "local://test.txt",
        "mime_type": "text/plain",
        "size_bytes": 100,
        "checksum": None,
        "is_deleted": False,
        "deleted_at": None,
        "metadata": {},
    }
    defaults.update(overrides)
    return WorkspaceFile(**defaults)


def _make_file_mock(**overrides) -> MagicMock:
    """Create a mock WorkspaceFile to simulate DB records with null fields."""
    defaults = {
        "id": "file-001",
        "name": "test.txt",
        "path": "/test.txt",
        "mime_type": "text/plain",
        "size_bytes": 100,
        "metadata": {},
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    mock = MagicMock(spec=WorkspaceFile)
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


# -- list_workspace_files tests --


@patch("backend.api.features.workspace.routes.get_or_create_workspace")
@patch("backend.api.features.workspace.routes.WorkspaceManager")
def test_list_files_returns_all_when_no_session(mock_manager_cls, mock_get_workspace):
    mock_get_workspace.return_value = _make_workspace()
    files = [
        _make_file(id="f1", name="a.txt", metadata={"origin": "user-upload"}),
        _make_file(id="f2", name="b.csv", metadata={"origin": "agent-created"}),
    ]
    mock_instance = AsyncMock()
    mock_instance.list_files.return_value = files
    mock_manager_cls.return_value = mock_instance

    response = client.get("/files")
    assert response.status_code == 200

    data = response.json()
    assert len(data["files"]) == 2
    assert data["has_more"] is False
    assert data["offset"] == 0
    assert data["files"][0]["id"] == "f1"
    assert data["files"][0]["metadata"] == {"origin": "user-upload"}
    assert data["files"][1]["id"] == "f2"
    mock_instance.list_files.assert_called_once_with(
        limit=201, offset=0, include_all_sessions=True
    )


@patch("backend.api.features.workspace.routes.get_or_create_workspace")
@patch("backend.api.features.workspace.routes.WorkspaceManager")
def test_list_files_scopes_to_session_when_provided(
    mock_manager_cls, mock_get_workspace, test_user_id
):
    mock_get_workspace.return_value = _make_workspace(user_id=test_user_id)
    mock_instance = AsyncMock()
    mock_instance.list_files.return_value = []
    mock_manager_cls.return_value = mock_instance

    response = client.get("/files?session_id=sess-123")
    assert response.status_code == 200

    data = response.json()
    assert data["files"] == []
    assert data["has_more"] is False
    mock_manager_cls.assert_called_once_with(test_user_id, "ws-001", "sess-123")
    mock_instance.list_files.assert_called_once_with(
        limit=201, offset=0, include_all_sessions=False
    )


@patch("backend.api.features.workspace.routes.get_or_create_workspace")
@patch("backend.api.features.workspace.routes.WorkspaceManager")
def test_list_files_null_metadata_coerced_to_empty_dict(
    mock_manager_cls, mock_get_workspace
):
    """Route uses `f.metadata or {}` for pre-existing files with null metadata."""
    mock_get_workspace.return_value = _make_workspace()
    mock_instance = AsyncMock()
    mock_instance.list_files.return_value = [_make_file_mock(metadata=None)]
    mock_manager_cls.return_value = mock_instance

    response = client.get("/files")
    assert response.status_code == 200
    assert response.json()["files"][0]["metadata"] == {}


# -- upload_file metadata tests --


@patch("backend.api.features.workspace.routes.get_or_create_workspace")
@patch("backend.api.features.workspace.routes.get_workspace_total_size")
@patch("backend.api.features.workspace.routes.scan_content_safe")
@patch("backend.api.features.workspace.routes.WorkspaceManager")
def test_upload_passes_user_upload_origin_metadata(
    mock_manager_cls, mock_scan, mock_total_size, mock_get_workspace
):
    mock_get_workspace.return_value = _make_workspace()
    mock_total_size.return_value = 100
    written = _make_file(id="new-file", name="doc.pdf")
    mock_instance = AsyncMock()
    mock_instance.write_file.return_value = written
    mock_manager_cls.return_value = mock_instance

    response = client.post(
        "/files/upload",
        files={"file": ("doc.pdf", b"fake-pdf-content", "application/pdf")},
    )
    assert response.status_code == 200

    mock_instance.write_file.assert_called_once()
    call_kwargs = mock_instance.write_file.call_args
    assert call_kwargs.kwargs.get("metadata") == {"origin": "user-upload"}


@patch("backend.api.features.workspace.routes.get_or_create_workspace")
@patch("backend.api.features.workspace.routes.get_workspace_total_size")
@patch("backend.api.features.workspace.routes.scan_content_safe")
@patch("backend.api.features.workspace.routes.WorkspaceManager")
def test_upload_returns_409_on_file_conflict(
    mock_manager_cls, mock_scan, mock_total_size, mock_get_workspace
):
    mock_get_workspace.return_value = _make_workspace()
    mock_total_size.return_value = 100
    mock_instance = AsyncMock()
    mock_instance.write_file.side_effect = ValueError("File already exists at path")
    mock_manager_cls.return_value = mock_instance

    response = client.post(
        "/files/upload",
        files={"file": ("dup.txt", b"content", "text/plain")},
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


# -- Restored upload/download/delete security + invariant tests --


def _upload(
    filename: str = "hello.txt",
    content: bytes = b"Hello, world!",
    content_type: str = "text/plain",
):
    return client.post(
        "/files/upload?session_id=sess-1",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )


_MOCK_FILE = WorkspaceFile(
    id="file-aaa-bbb",
    workspace_id="ws-001",
    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    name="hello.txt",
    path="/sessions/sess-1/hello.txt",
    mime_type="text/plain",
    size_bytes=13,
    storage_path="local://hello.txt",
)


def test_upload_happy_path(mocker):
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=_make_workspace(),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size",
        return_value=0,
    )
    mocker.patch(
        "backend.api.features.workspace.routes.scan_content_safe",
        return_value=None,
    )
    mock_manager = mocker.MagicMock()
    mock_manager.write_file = mocker.AsyncMock(return_value=_MOCK_FILE)
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = _upload()
    assert response.status_code == 200
    data = response.json()
    assert data["file_id"] == "file-aaa-bbb"
    assert data["name"] == "hello.txt"
    assert data["size_bytes"] == 13


def test_upload_exceeds_max_file_size(mocker):
    """Files larger than max_file_size_mb should be rejected with 413."""
    cfg = mocker.patch("backend.api.features.workspace.routes.Config")
    cfg.return_value.max_file_size_mb = 0  # 0 MB → any content is too big
    cfg.return_value.max_workspace_storage_mb = 500

    response = _upload(content=b"x" * 1024)
    assert response.status_code == 413


def test_upload_storage_quota_exceeded(mocker):
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=_make_workspace(),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size",
        return_value=500 * 1024 * 1024,
    )

    response = _upload()
    assert response.status_code == 413
    assert "Storage limit exceeded" in response.text


def test_upload_post_write_quota_race(mocker):
    """Concurrent upload tipping over limit after write should soft-delete + 413."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=_make_workspace(),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size",
        side_effect=[0, 600 * 1024 * 1024],
    )
    mocker.patch(
        "backend.api.features.workspace.routes.scan_content_safe",
        return_value=None,
    )
    mock_manager = mocker.MagicMock()
    mock_manager.write_file = mocker.AsyncMock(return_value=_MOCK_FILE)
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )
    mock_delete = mocker.patch(
        "backend.api.features.workspace.routes.soft_delete_workspace_file",
        return_value=None,
    )

    response = _upload()
    assert response.status_code == 413
    mock_delete.assert_called_once_with("file-aaa-bbb", "ws-001")


def test_upload_any_extension(mocker):
    """Any file extension should be accepted — ClamAV is the security layer."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=_make_workspace(),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size",
        return_value=0,
    )
    mocker.patch(
        "backend.api.features.workspace.routes.scan_content_safe",
        return_value=None,
    )
    mock_manager = mocker.MagicMock()
    mock_manager.write_file = mocker.AsyncMock(return_value=_MOCK_FILE)
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = _upload(filename="data.xyz", content=b"arbitrary")
    assert response.status_code == 200


def test_upload_blocked_by_virus_scan(mocker):
    """Files flagged by ClamAV should be rejected and never written to storage."""
    from backend.api.features.store.exceptions import VirusDetectedError

    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=_make_workspace(),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size",
        return_value=0,
    )
    mocker.patch(
        "backend.api.features.workspace.routes.scan_content_safe",
        side_effect=VirusDetectedError("Eicar-Test-Signature"),
    )
    mock_manager = mocker.MagicMock()
    mock_manager.write_file = mocker.AsyncMock(return_value=_MOCK_FILE)
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = _upload(filename="evil.exe", content=b"X5O!P%@AP...")
    assert response.status_code == 400
    mock_manager.write_file.assert_not_called()


def test_upload_file_without_extension(mocker):
    """Files without an extension should be accepted and stored as-is."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=_make_workspace(),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size",
        return_value=0,
    )
    mocker.patch(
        "backend.api.features.workspace.routes.scan_content_safe",
        return_value=None,
    )
    mock_manager = mocker.MagicMock()
    mock_manager.write_file = mocker.AsyncMock(return_value=_MOCK_FILE)
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = _upload(
        filename="Makefile",
        content=b"all:\n\techo hello",
        content_type="application/octet-stream",
    )
    assert response.status_code == 200
    mock_manager.write_file.assert_called_once()
    assert mock_manager.write_file.call_args[0][1] == "Makefile"


def test_upload_strips_path_components(mocker):
    """Path-traversal filenames should be reduced to their basename."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=_make_workspace(),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size",
        return_value=0,
    )
    mocker.patch(
        "backend.api.features.workspace.routes.scan_content_safe",
        return_value=None,
    )
    mock_manager = mocker.MagicMock()
    mock_manager.write_file = mocker.AsyncMock(return_value=_MOCK_FILE)
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    _upload(filename="../../etc/passwd.txt")

    mock_manager.write_file.assert_called_once()
    call_args = mock_manager.write_file.call_args
    assert call_args[0][1] == "passwd.txt"


def test_download_file_not_found(mocker):
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        return_value=_make_workspace(),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_file",
        return_value=None,
    )

    response = client.get("/files/some-file-id/download")
    assert response.status_code == 404


def test_delete_file_success(mocker):
    """Deleting an existing file should return {"deleted": true}."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        return_value=_make_workspace(),
    )
    mock_manager = mocker.MagicMock()
    mock_manager.delete_file = mocker.AsyncMock(return_value=True)
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = client.delete("/files/file-aaa-bbb")
    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    mock_manager.delete_file.assert_called_once_with("file-aaa-bbb")


def test_delete_file_not_found(mocker):
    """Deleting a non-existent file should return 404."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        return_value=_make_workspace(),
    )
    mock_manager = mocker.MagicMock()
    mock_manager.delete_file = mocker.AsyncMock(return_value=False)
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = client.delete("/files/nonexistent-id")
    assert response.status_code == 404
    assert "File not found" in response.text


def test_delete_file_no_workspace(mocker):
    """Deleting when user has no workspace should return 404."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        return_value=None,
    )

    response = client.delete("/files/file-aaa-bbb")
    assert response.status_code == 404
    assert "Workspace not found" in response.text


def test_upload_write_file_too_large_returns_413(mocker):
    """write_file raises ValueError("File too large: …") → must map to 413."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=_make_workspace(),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size",
        return_value=0,
    )
    mocker.patch(
        "backend.api.features.workspace.routes.scan_content_safe",
        return_value=None,
    )
    mock_manager = mocker.MagicMock()
    mock_manager.write_file = mocker.AsyncMock(
        side_effect=ValueError("File too large: 900 bytes exceeds 1MB limit")
    )
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = _upload()
    assert response.status_code == 413
    assert "File too large" in response.text


def test_upload_write_file_conflict_returns_409(mocker):
    """Non-'File too large' ValueErrors from write_file stay as 409."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=_make_workspace(),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size",
        return_value=0,
    )
    mocker.patch(
        "backend.api.features.workspace.routes.scan_content_safe",
        return_value=None,
    )
    mock_manager = mocker.MagicMock()
    mock_manager.write_file = mocker.AsyncMock(
        side_effect=ValueError("File already exists at path: /sessions/x/a.txt")
    )
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = _upload()
    assert response.status_code == 409
    assert "already exists" in response.text


@patch("backend.api.features.workspace.routes.get_or_create_workspace")
@patch("backend.api.features.workspace.routes.WorkspaceManager")
def test_list_files_has_more_true_when_limit_exceeded(
    mock_manager_cls, mock_get_workspace
):
    """The limit+1 fetch trick must flip has_more=True and trim the page."""
    mock_get_workspace.return_value = _make_workspace()
    # Backend was asked for limit+1=3, and returned exactly 3 items.
    files = [
        _make_file(id="f1", name="a.txt"),
        _make_file(id="f2", name="b.txt"),
        _make_file(id="f3", name="c.txt"),
    ]
    mock_instance = AsyncMock()
    mock_instance.list_files.return_value = files
    mock_manager_cls.return_value = mock_instance

    response = client.get("/files?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert data["has_more"] is True
    assert len(data["files"]) == 2
    assert data["files"][0]["id"] == "f1"
    assert data["files"][1]["id"] == "f2"
    mock_instance.list_files.assert_called_once_with(
        limit=3, offset=0, include_all_sessions=True
    )


@patch("backend.api.features.workspace.routes.get_or_create_workspace")
@patch("backend.api.features.workspace.routes.WorkspaceManager")
def test_list_files_has_more_false_when_exactly_page_size(
    mock_manager_cls, mock_get_workspace
):
    """Exactly `limit` rows means we're on the last page — has_more=False."""
    mock_get_workspace.return_value = _make_workspace()
    files = [_make_file(id="f1", name="a.txt"), _make_file(id="f2", name="b.txt")]
    mock_instance = AsyncMock()
    mock_instance.list_files.return_value = files
    mock_manager_cls.return_value = mock_instance

    response = client.get("/files?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert data["has_more"] is False
    assert len(data["files"]) == 2


@patch("backend.api.features.workspace.routes.get_or_create_workspace")
@patch("backend.api.features.workspace.routes.WorkspaceManager")
def test_list_files_offset_is_echoed_back(mock_manager_cls, mock_get_workspace):
    mock_get_workspace.return_value = _make_workspace()
    mock_instance = AsyncMock()
    mock_instance.list_files.return_value = []
    mock_manager_cls.return_value = mock_instance

    response = client.get("/files?offset=50&limit=10")
    assert response.status_code == 200
    assert response.json()["offset"] == 50
    mock_instance.list_files.assert_called_once_with(
        limit=11, offset=50, include_all_sessions=True
    )


# -- _sanitize_filename_for_header tests --


class TestSanitizeFilenameForHeader:
    def test_simple_ascii_attachment(self):
        from backend.api.features.workspace.routes import _sanitize_filename_for_header

        assert _sanitize_filename_for_header("report.pdf") == (
            'attachment; filename="report.pdf"'
        )

    def test_inline_disposition(self):
        from backend.api.features.workspace.routes import _sanitize_filename_for_header

        assert _sanitize_filename_for_header("image.png", disposition="inline") == (
            'inline; filename="image.png"'
        )

    def test_strips_cr_lf_null(self):
        from backend.api.features.workspace.routes import _sanitize_filename_for_header

        result = _sanitize_filename_for_header("a\rb\nc\x00d.txt")
        assert "\r" not in result
        assert "\n" not in result
        assert "\x00" not in result
        assert 'filename="abcd.txt"' in result

    def test_escapes_quotes(self):
        from backend.api.features.workspace.routes import _sanitize_filename_for_header

        result = _sanitize_filename_for_header('file"name.txt')
        assert 'filename="file\\"name.txt"' in result

    def test_header_injection_blocked(self):
        from backend.api.features.workspace.routes import _sanitize_filename_for_header

        result = _sanitize_filename_for_header("evil.txt\r\nX-Injected: true")
        # CR/LF stripped — the remaining text is safely inside the quoted value
        assert "\r" not in result
        assert "\n" not in result
        assert result == 'attachment; filename="evil.txtX-Injected: true"'

    def test_unicode_uses_rfc5987(self):
        from backend.api.features.workspace.routes import _sanitize_filename_for_header

        result = _sanitize_filename_for_header("日本語.pdf")
        assert "filename*=UTF-8''" in result
        assert "attachment" in result

    def test_unicode_inline(self):
        from backend.api.features.workspace.routes import _sanitize_filename_for_header

        result = _sanitize_filename_for_header("图片.png", disposition="inline")
        assert result.startswith("inline; filename*=UTF-8''")

    def test_empty_filename(self):
        from backend.api.features.workspace.routes import _sanitize_filename_for_header

        result = _sanitize_filename_for_header("")
        assert result == 'attachment; filename=""'


# -- _create_streaming_response tests --


class TestCreateStreamingResponse:
    def test_attachment_disposition_by_default(self):
        from backend.api.features.workspace.routes import _create_streaming_response

        file = _make_file(name="data.bin", mime_type="application/octet-stream")
        response = _create_streaming_response(b"binary-data", file)
        assert (
            response.headers["Content-Disposition"] == 'attachment; filename="data.bin"'
        )
        assert response.headers["Content-Type"] == "application/octet-stream"
        assert response.headers["Content-Length"] == "11"
        assert response.body == b"binary-data"

    def test_inline_disposition(self):
        from backend.api.features.workspace.routes import _create_streaming_response

        file = _make_file(name="photo.png", mime_type="image/png")
        response = _create_streaming_response(b"\x89PNG", file, inline=True)
        assert response.headers["Content-Disposition"] == 'inline; filename="photo.png"'
        assert response.headers["Content-Type"] == "image/png"

    def test_inline_sanitizes_filename(self):
        from backend.api.features.workspace.routes import _create_streaming_response

        file = _make_file(name='evil"\r\n.txt', mime_type="text/plain")
        response = _create_streaming_response(b"data", file, inline=True)
        assert "\r" not in response.headers["Content-Disposition"]
        assert "\n" not in response.headers["Content-Disposition"]
        assert "inline" in response.headers["Content-Disposition"]

    def test_content_length_matches_body(self):
        from backend.api.features.workspace.routes import _create_streaming_response

        content = b"x" * 1000
        file = _make_file(name="big.bin", mime_type="application/octet-stream")
        response = _create_streaming_response(content, file)
        assert response.headers["Content-Length"] == "1000"


# -- create_file_download_response tests --


class TestCreateFileDownloadResponse:
    @pytest.mark.asyncio
    async def test_local_storage_returns_streaming_response(self, mocker):
        from backend.api.features.workspace.routes import create_file_download_response

        mock_storage = AsyncMock()
        mock_storage.retrieve.return_value = b"file contents"
        mocker.patch(
            "backend.api.features.workspace.routes.get_workspace_storage",
            return_value=mock_storage,
        )

        file = _make_file(
            storage_path="local://uploads/test.txt",
            mime_type="text/plain",
        )
        response = await create_file_download_response(file)
        assert response.status_code == 200
        assert response.body == b"file contents"
        assert "attachment" in response.headers["Content-Disposition"]

    @pytest.mark.asyncio
    async def test_local_storage_inline(self, mocker):
        from backend.api.features.workspace.routes import create_file_download_response

        mock_storage = AsyncMock()
        mock_storage.retrieve.return_value = b"\x89PNG"
        mocker.patch(
            "backend.api.features.workspace.routes.get_workspace_storage",
            return_value=mock_storage,
        )

        file = _make_file(
            storage_path="local://uploads/photo.png",
            mime_type="image/png",
            name="photo.png",
        )
        response = await create_file_download_response(file, inline=True)
        assert "inline" in response.headers["Content-Disposition"]

    @pytest.mark.asyncio
    async def test_gcs_redirect(self, mocker):
        from backend.api.features.workspace.routes import create_file_download_response

        mock_storage = AsyncMock()
        mock_storage.get_download_url.return_value = (
            "https://storage.googleapis.com/signed-url"
        )
        mocker.patch(
            "backend.api.features.workspace.routes.get_workspace_storage",
            return_value=mock_storage,
        )

        file = _make_file(storage_path="gcs://bucket/file.pdf")
        response = await create_file_download_response(file)
        assert response.status_code == 302
        assert (
            response.headers["location"] == "https://storage.googleapis.com/signed-url"
        )

    @pytest.mark.asyncio
    async def test_gcs_api_fallback_streams_directly(self, mocker):
        from backend.api.features.workspace.routes import create_file_download_response

        mock_storage = AsyncMock()
        mock_storage.get_download_url.return_value = "/api/fallback"
        mock_storage.retrieve.return_value = b"fallback content"
        mocker.patch(
            "backend.api.features.workspace.routes.get_workspace_storage",
            return_value=mock_storage,
        )

        file = _make_file(storage_path="gcs://bucket/file.txt")
        response = await create_file_download_response(file)
        assert response.status_code == 200
        assert response.body == b"fallback content"

    @pytest.mark.asyncio
    async def test_gcs_signed_url_failure_falls_back_to_streaming(self, mocker):
        from backend.api.features.workspace.routes import create_file_download_response

        mock_storage = AsyncMock()
        mock_storage.get_download_url.side_effect = RuntimeError("GCS error")
        mock_storage.retrieve.return_value = b"streamed"
        mocker.patch(
            "backend.api.features.workspace.routes.get_workspace_storage",
            return_value=mock_storage,
        )

        file = _make_file(storage_path="gcs://bucket/file.txt")
        response = await create_file_download_response(file)
        assert response.status_code == 200
        assert response.body == b"streamed"

    @pytest.mark.asyncio
    async def test_gcs_total_failure_raises(self, mocker):
        from backend.api.features.workspace.routes import create_file_download_response

        mock_storage = AsyncMock()
        mock_storage.get_download_url.side_effect = RuntimeError("GCS error")
        mock_storage.retrieve.side_effect = RuntimeError("Also failed")
        mocker.patch(
            "backend.api.features.workspace.routes.get_workspace_storage",
            return_value=mock_storage,
        )

        file = _make_file(storage_path="gcs://bucket/file.txt")
        with pytest.raises(RuntimeError, match="Also failed"):
            await create_file_download_response(file)

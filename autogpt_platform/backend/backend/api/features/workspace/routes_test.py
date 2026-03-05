"""Tests for workspace file upload and download routes."""

import io
from datetime import datetime, timezone

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.api.features.workspace import routes as workspace_routes
from backend.data.workspace import WorkspaceFile

app = fastapi.FastAPI()
app.include_router(workspace_routes.router)


@app.exception_handler(ValueError)
async def _value_error_handler(
    request: fastapi.Request, exc: ValueError
) -> fastapi.responses.JSONResponse:
    """Mirror the production ValueError → 400 mapping from rest_api.py."""
    return fastapi.responses.JSONResponse(status_code=400, content={"detail": str(exc)})


client = fastapi.testclient.TestClient(app)

TEST_USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"

MOCK_WORKSPACE = type("W", (), {"id": "ws-1"})()

_NOW = datetime(2023, 1, 1, tzinfo=timezone.utc)

MOCK_FILE = WorkspaceFile(
    id="file-aaa-bbb",
    workspace_id="ws-1",
    created_at=_NOW,
    updated_at=_NOW,
    name="hello.txt",
    path="/session/hello.txt",
    mime_type="text/plain",
    size_bytes=13,
    storage_path="local://hello.txt",
)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _upload(
    filename: str = "hello.txt",
    content: bytes = b"Hello, world!",
    content_type: str = "text/plain",
):
    """Helper to POST a file upload."""
    return client.post(
        "/files/upload?session_id=sess-1",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )


# ---- Happy path ----


def test_upload_happy_path(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=MOCK_WORKSPACE,
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
    mock_manager.write_file = mocker.AsyncMock(return_value=MOCK_FILE)
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


# ---- Per-file size limit ----


def test_upload_exceeds_max_file_size(mocker: pytest_mock.MockFixture):
    """Files larger than max_file_size_mb should be rejected with 413."""
    cfg = mocker.patch("backend.api.features.workspace.routes.Config")
    cfg.return_value.max_file_size_mb = 0  # 0 MB → any content is too big
    cfg.return_value.max_workspace_storage_mb = 500

    response = _upload(content=b"x" * 1024)
    assert response.status_code == 413


# ---- Storage quota exceeded ----


def test_upload_storage_quota_exceeded(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=MOCK_WORKSPACE,
    )
    # Current usage already at limit
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size",
        return_value=500 * 1024 * 1024,
    )

    response = _upload()
    assert response.status_code == 413
    assert "Storage limit exceeded" in response.text


# ---- Post-write quota race (B2) ----


def test_upload_post_write_quota_race(mocker: pytest_mock.MockFixture):
    """If a concurrent upload tips the total over the limit after write,
    the file should be soft-deleted and 413 returned."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=MOCK_WORKSPACE,
    )
    # Pre-write check passes (under limit), but post-write check fails
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size",
        side_effect=[0, 600 * 1024 * 1024],  # first call OK, second over limit
    )
    mocker.patch(
        "backend.api.features.workspace.routes.scan_content_safe",
        return_value=None,
    )
    mock_manager = mocker.MagicMock()
    mock_manager.write_file = mocker.AsyncMock(return_value=MOCK_FILE)
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
    mock_delete.assert_called_once_with("file-aaa-bbb", "ws-1")


# ---- Any extension accepted (no allowlist) ----


def test_upload_any_extension(mocker: pytest_mock.MockFixture):
    """Any file extension should be accepted — ClamAV is the security layer."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=MOCK_WORKSPACE,
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
    mock_manager.write_file = mocker.AsyncMock(return_value=MOCK_FILE)
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = _upload(filename="data.xyz", content=b"arbitrary")
    assert response.status_code == 200


# ---- Virus scan rejection ----


def test_upload_blocked_by_virus_scan(mocker: pytest_mock.MockFixture):
    """Files flagged by ClamAV should be rejected and never written to storage."""
    from backend.api.features.store.exceptions import VirusDetectedError

    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=MOCK_WORKSPACE,
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
    mock_manager.write_file = mocker.AsyncMock(return_value=MOCK_FILE)
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = _upload(filename="evil.exe", content=b"X5O!P%@AP...")
    assert response.status_code == 400
    assert "Virus detected" in response.text
    mock_manager.write_file.assert_not_called()


# ---- No file extension ----


def test_upload_file_without_extension(mocker: pytest_mock.MockFixture):
    """Files without an extension should be accepted and stored as-is."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=MOCK_WORKSPACE,
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
    mock_manager.write_file = mocker.AsyncMock(return_value=MOCK_FILE)
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


# ---- Filename sanitization (SF5) ----


def test_upload_strips_path_components(mocker: pytest_mock.MockFixture):
    """Path-traversal filenames should be reduced to their basename."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=MOCK_WORKSPACE,
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
    mock_manager.write_file = mocker.AsyncMock(return_value=MOCK_FILE)
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    # Filename with traversal
    _upload(filename="../../etc/passwd.txt")

    # write_file should have been called with just the basename
    mock_manager.write_file.assert_called_once()
    call_args = mock_manager.write_file.call_args
    assert call_args[0][1] == "passwd.txt"


# ---- Download ----


def test_download_file_not_found(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        return_value=MOCK_WORKSPACE,
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_file",
        return_value=None,
    )

    response = client.get("/files/some-file-id/download")
    assert response.status_code == 404


# ---- Delete ----


def test_delete_file_success(mocker: pytest_mock.MockFixture):
    """Deleting an existing file should return {"deleted": true}."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        return_value=MOCK_WORKSPACE,
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


def test_delete_file_not_found(mocker: pytest_mock.MockFixture):
    """Deleting a non-existent file should return 404."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        return_value=MOCK_WORKSPACE,
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


def test_delete_file_no_workspace(mocker: pytest_mock.MockFixture):
    """Deleting when user has no workspace should return 404."""
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        return_value=None,
    )

    response = client.delete("/files/file-aaa-bbb")
    assert response.status_code == 404
    assert "Workspace not found" in response.text

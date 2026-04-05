from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.workspace.routes import router
from backend.data.workspace import Workspace, WorkspaceFile

app = fastapi.FastAPI()
app.include_router(router)
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
    assert data["total_count"] == 2
    assert data["files"][0]["id"] == "f1"
    assert data["files"][0]["metadata"] == {"origin": "user-upload"}
    assert data["files"][1]["id"] == "f2"
    mock_instance.list_files.assert_called_once_with(include_all_sessions=True)


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
    assert data["total_count"] == 0
    assert data["files"] == []
    mock_manager_cls.assert_called_once_with(test_user_id, "ws-001", "sess-123")
    mock_instance.list_files.assert_called_once_with(include_all_sessions=False)


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

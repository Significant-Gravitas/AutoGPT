from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

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


def _make_workspace() -> Workspace:
    return Workspace(
        id="ws-001",
        user_id="test-user-id",
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


@patch("backend.api.features.workspace.routes.get_or_create_workspace")
@patch("backend.api.features.workspace.routes.WorkspaceManager")
def test_list_files_returns_all_files(mock_manager_cls, mock_get_workspace):
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
def test_list_files_with_session_id(mock_manager_cls, mock_get_workspace):
    mock_get_workspace.return_value = _make_workspace()
    mock_instance = AsyncMock()
    mock_instance.list_files.return_value = []
    mock_manager_cls.return_value = mock_instance

    response = client.get("/files?session_id=sess-123")
    assert response.status_code == 200

    data = response.json()
    assert data["total_count"] == 0
    assert data["files"] == []
    mock_manager_cls.assert_called_once_with(
        "3e53486c-cf57-477e-ba2a-cb02dc828e1a", "ws-001", "sess-123"
    )
    mock_instance.list_files.assert_called_once_with(include_all_sessions=False)


@patch("backend.api.features.workspace.routes.get_or_create_workspace")
@patch("backend.api.features.workspace.routes.WorkspaceManager")
def test_list_files_null_metadata_defaults_to_empty_dict(
    mock_manager_cls, mock_get_workspace
):
    mock_get_workspace.return_value = _make_workspace()
    mock_instance = AsyncMock()
    file_with_null_metadata = _make_file()
    file_with_null_metadata.metadata = None  # type: ignore[assignment]
    mock_instance.list_files.return_value = [file_with_null_metadata]
    mock_manager_cls.return_value = mock_instance

    response = client.get("/files")
    assert response.status_code == 200
    assert response.json()["files"][0]["metadata"] == {}

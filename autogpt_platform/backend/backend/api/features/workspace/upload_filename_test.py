import io
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.features.workspace.routes import router
from backend.data.workspace import Workspace, WorkspaceFile
from backend.util.workspace_storage import LocalWorkspaceStorage


@pytest.fixture
def client(mock_jwt_user):
    app = fastapi.FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    return fastapi.testclient.TestClient(app)


@pytest.fixture
def workspace(mocker):
    now = datetime.now(timezone.utc)
    workspace = Workspace(
        id="ws-001", user_id="test-user-id", created_at=now, updated_at=now
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        return_value=workspace,
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_total_size", return_value=0
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_storage_limit_bytes",
        return_value=250 * 1024 * 1024,
    )
    return workspace


def test_upload_multibyte_name_preserves_metadata_and_content(
    client, workspace, mocker, tmp_path
):
    filename = "夜" * 86 + ".txt"
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path))
    mocker.patch("backend.util.workspace.get_workspace_storage", return_value=storage)
    mocker.patch(
        "backend.util.workspace.get_workspace_storage_limit_bytes",
        return_value=250 * 1024 * 1024,
    )
    mocker.patch("backend.util.workspace.scan_content_safe", new=AsyncMock())
    db = MagicMock()
    db.get_workspace_total_size = AsyncMock(return_value=0)
    db.get_workspace_file_by_path = AsyncMock(return_value=None)

    async def create_file(**kwargs):
        return WorkspaceFile(
            id=kwargs["file_id"],
            workspace_id=workspace.id,
            name=kwargs["name"],
            path=kwargs["path"],
            storage_path=kwargs["storage_path"],
            mime_type=kwargs["mime_type"],
            size_bytes=kwargs["size_bytes"],
            checksum=kwargs["checksum"],
            metadata=kwargs["metadata"],
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
        )

    db.create_workspace_file = AsyncMock(side_effect=create_file)
    mocker.patch("backend.util.workspace.workspace_db", return_value=db)

    response = client.post(
        "/files/upload", files={"file": (filename, io.BytesIO(b"hello"), "text/plain")}
    )

    assert response.status_code == 200, response.text
    assert response.json()["name"] == filename
    assert response.json()["path"] == f"/{filename}"
    stored = db.create_workspace_file.await_args.kwargs["storage_path"]
    assert storage._parse_storage_path(stored).read_bytes() == b"hello"


def test_upload_storage_error_hides_exception_details(client, workspace, mocker):
    secret = "/data/workspaces/ws-001/f1/secret-name.txt"
    manager = MagicMock()
    manager.write_file = AsyncMock(
        side_effect=OSError(36, "File name too long", secret)
    )
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager", return_value=manager
    )

    response = client.post(
        "/files/upload", files={"file": ("secret-name.txt", b"x", "text/plain")}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to store file. Please try again."
    assert "/data/" not in response.text
    assert "File name too long" not in response.text


def test_upload_conflict_retains_useful_message(client, workspace, mocker):
    manager = MagicMock()
    manager.write_file = AsyncMock(side_effect=ValueError("File already exists"))
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager", return_value=manager
    )

    response = client.post(
        "/files/upload", files={"file": ("existing.txt", b"x", "text/plain")}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "File already exists"

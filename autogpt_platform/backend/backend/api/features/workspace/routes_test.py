import datetime
import io
import json
import uuid as uuid_mod
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from pytest_mock import MockFixture
from pytest_snapshot.plugin import Snapshot

from backend.api.features.workspace.routes import router
from backend.data.workspace import Workspace, WorkspaceFile

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

FIXED_NOW = datetime.datetime(2023, 1, 1, 0, 0, 0)
WORKSPACE_ID = "ws-test-1234"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module"""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _make_workspace(user_id: str) -> Workspace:
    return Workspace(
        id=WORKSPACE_ID,
        user_id=user_id,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )


def _make_workspace_file(
    *,
    file_id: str = "file-abc-123",
    name: str = "test.png",
    mime_type: str = "image/png",
    size_bytes: int = 1024,
) -> WorkspaceFile:
    return WorkspaceFile(
        id=file_id,
        workspace_id=WORKSPACE_ID,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
        name=name,
        path=f"/builder-uploads/{file_id}/{name}",
        storage_path=f"local://{WORKSPACE_ID}/{file_id}/{name}",
        mime_type=mime_type,
        size_bytes=size_bytes,
    )


# ── DELETE /files/{file_id} ─────────────────────────────────────────────


def test_delete_file_workspace_not_found(mocker: MockFixture, test_user_id: str):
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.delete("/files/file-abc-123")

    assert response.status_code == 404
    assert response.json()["detail"] == "Workspace not found"


def test_delete_file_not_found(mocker: MockFixture, test_user_id: str):
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        new_callable=AsyncMock,
        return_value=_make_workspace(test_user_id),
    )

    mock_manager = AsyncMock()
    mock_manager.delete_file.return_value = False
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = client.delete("/files/file-abc-123")

    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"
    mock_manager.delete_file.assert_awaited_once_with("file-abc-123")


def test_delete_file_success(
    mocker: MockFixture, test_user_id: str, snapshot: Snapshot
):
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        new_callable=AsyncMock,
        return_value=_make_workspace(test_user_id),
    )

    mock_manager = AsyncMock()
    mock_manager.delete_file.return_value = True
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    response = client.delete("/files/file-abc-123")

    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    mock_manager.delete_file.assert_awaited_once_with("file-abc-123")

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response.json(), indent=2, sort_keys=True),
        "delete_file_success",
    )


# ── POST /files/upload ──────────────────────────────────────────────────


def test_upload_file_success(
    mocker: MockFixture, test_user_id: str, snapshot: Snapshot
):
    workspace = _make_workspace(test_user_id)
    workspace_file = _make_workspace_file()

    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        new_callable=AsyncMock,
        return_value=workspace,
    )

    mock_manager = AsyncMock()
    mock_manager.write_file.return_value = workspace_file
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    # Deterministic uuid so the path is predictable
    mocker.patch(
        "backend.api.features.workspace.routes.uuid.uuid4",
        return_value=uuid_mod.UUID("12345678-1234-1234-1234-123456781234"),
    )

    file_content = b"fake image data"
    response = client.post(
        "/files/upload",
        files={"file": ("test.png", io.BytesIO(file_content), "image/png")},
    )

    assert response.status_code == 200

    data = response.json()
    assert (
        data["file_uri"]
        == f"workspace://{workspace_file.id}#{workspace_file.mime_type}"
    )
    assert data["file_name"] == workspace_file.name
    assert data["size"] == workspace_file.size_bytes
    assert data["content_type"] == workspace_file.mime_type

    mock_manager.write_file.assert_awaited_once_with(
        content=file_content,
        filename="test.png",
        path="/builder-uploads/12345678-1234-1234-1234-123456781234/test.png",
        mime_type="image/png",
    )

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True),
        "upload_file_success",
    )


def test_upload_file_default_filename_and_content_type(
    mocker: MockFixture, test_user_id: str
):
    """Upload without explicit filename/content_type uses defaults."""
    workspace = _make_workspace(test_user_id)
    workspace_file = _make_workspace_file(
        name="upload", mime_type="application/octet-stream"
    )

    mocker.patch(
        "backend.api.features.workspace.routes.get_or_create_workspace",
        new_callable=AsyncMock,
        return_value=workspace,
    )

    mock_manager = AsyncMock()
    mock_manager.write_file.return_value = workspace_file
    mocker.patch(
        "backend.api.features.workspace.routes.WorkspaceManager",
        return_value=mock_manager,
    )

    mocker.patch(
        "backend.api.features.workspace.routes.uuid.uuid4",
        return_value=uuid_mod.UUID("56785678-5678-5678-5678-567856785678"),
    )

    file_content = b"some bytes"
    response = client.post(
        "/files/upload",
        files={
            "file": ("upload", io.BytesIO(file_content), "application/octet-stream")
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["content_type"] == "application/octet-stream"
    assert data["file_name"] == "upload"

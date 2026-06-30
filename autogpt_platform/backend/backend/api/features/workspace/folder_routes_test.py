from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.library.exceptions import FolderAlreadyExistsError
from backend.api.features.workspace.folder_routes import router
from backend.data.workspace import Workspace, WorkspaceFile
from backend.data.workspace_folder import WorkspaceFolder
from backend.util.exceptions import NotFoundError

app = fastapi.FastAPI()
# The /folders prefix is applied at the mount point (see rest_api.py), so apply
# it here too to exercise the same paths the app serves.
app.include_router(router, prefix="/folders")


@app.exception_handler(FolderAlreadyExistsError)
async def _folder_exists_handler(
    request: fastapi.Request, exc: FolderAlreadyExistsError
) -> fastapi.responses.JSONResponse:
    return fastapi.responses.JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(NotFoundError)
async def _not_found_handler(
    request: fastapi.Request, exc: NotFoundError
) -> fastapi.responses.JSONResponse:
    return fastapi.responses.JSONResponse(status_code=404, content={"detail": str(exc)})


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


def _make_folder(**overrides) -> WorkspaceFolder:
    defaults = {
        "id": "fld-1",
        "workspace_id": "ws-001",
        "name": "Reports",
        "icon": None,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "file_count": 0,
    }
    defaults.update(overrides)
    return WorkspaceFolder(**defaults)


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
    }
    defaults.update(overrides)
    return WorkspaceFile(**defaults)


def test_list_folders_returns_folders(mocker):
    mocker.patch(
        "backend.api.features.workspace.folder_routes.get_or_create_workspace",
        AsyncMock(return_value=_make_workspace()),
    )
    mocker.patch(
        "backend.api.features.workspace.folder_routes.list_folders",
        AsyncMock(return_value=[_make_folder(file_count=3)]),
    )

    response = client.get("/folders")
    assert response.status_code == 200
    data = response.json()
    assert len(data["folders"]) == 1
    assert data["folders"][0]["id"] == "fld-1"
    assert data["folders"][0]["file_count"] == 3


def test_create_folder_success(mocker):
    mocker.patch(
        "backend.api.features.workspace.folder_routes.get_or_create_workspace",
        AsyncMock(return_value=_make_workspace()),
    )
    create = mocker.patch(
        "backend.api.features.workspace.folder_routes.create_folder",
        AsyncMock(return_value=_make_folder(name="Invoices")),
    )

    response = client.post("/folders", json={"name": "Invoices"})
    assert response.status_code == 201
    assert response.json()["name"] == "Invoices"
    create.assert_awaited_once_with(workspace_id="ws-001", name="Invoices", icon=None)


def test_create_folder_name_conflict_returns_409(mocker):
    mocker.patch(
        "backend.api.features.workspace.folder_routes.get_or_create_workspace",
        AsyncMock(return_value=_make_workspace()),
    )
    mocker.patch(
        "backend.api.features.workspace.folder_routes.create_folder",
        AsyncMock(side_effect=FolderAlreadyExistsError("exists")),
    )

    response = client.post("/folders", json={"name": "Reports"})
    assert response.status_code == 409


def test_update_folder_success(mocker):
    mocker.patch(
        "backend.api.features.workspace.folder_routes.get_or_create_workspace",
        AsyncMock(return_value=_make_workspace()),
    )
    update = mocker.patch(
        "backend.api.features.workspace.folder_routes.update_folder",
        AsyncMock(return_value=_make_folder(name="Renamed")),
    )

    response = client.patch("/folders/fld-1", json={"name": "Renamed"})
    assert response.status_code == 200
    assert response.json()["name"] == "Renamed"
    update.assert_awaited_once_with(
        folder_id="fld-1", workspace_id="ws-001", name="Renamed", icon=None
    )


def test_update_folder_not_found_returns_404(mocker):
    mocker.patch(
        "backend.api.features.workspace.folder_routes.get_or_create_workspace",
        AsyncMock(return_value=_make_workspace()),
    )
    mocker.patch(
        "backend.api.features.workspace.folder_routes.update_folder",
        AsyncMock(side_effect=NotFoundError("nope")),
    )

    response = client.patch("/folders/missing", json={"name": "X"})
    assert response.status_code == 404


def test_delete_folder_returns_204(mocker):
    mocker.patch(
        "backend.api.features.workspace.folder_routes.get_or_create_workspace",
        AsyncMock(return_value=_make_workspace()),
    )
    delete = mocker.patch(
        "backend.api.features.workspace.folder_routes.delete_folder",
        AsyncMock(return_value=None),
    )

    response = client.delete("/folders/fld-1")
    assert response.status_code == 204
    delete.assert_awaited_once_with(folder_id="fld-1", workspace_id="ws-001")


def test_bulk_move_files_returns_updated_files(mocker):
    mocker.patch(
        "backend.api.features.workspace.folder_routes.get_or_create_workspace",
        AsyncMock(return_value=_make_workspace()),
    )
    move = mocker.patch(
        "backend.api.features.workspace.folder_routes.bulk_move_files_to_folder",
        AsyncMock(return_value=[_make_file(id="f1", folder_id="fld-1")]),
    )

    response = client.post(
        "/folders/files/bulk-move",
        json={"file_ids": ["f1"], "folder_id": "fld-1"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data[0]["id"] == "f1"
    assert data[0]["folder_id"] == "fld-1"
    move.assert_awaited_once_with(
        workspace_id="ws-001", file_ids=["f1"], folder_id="fld-1"
    )


def test_bulk_move_rejects_empty_folder_id():
    """An empty-string folder_id is rejected at validation (422), not passed
    through to a foreign-key error."""
    response = client.post(
        "/folders/files/bulk-move",
        json={"file_ids": ["f1"], "folder_id": ""},
    )
    assert response.status_code == 422


def test_bulk_move_files_to_root(mocker):
    """folder_id omitted means move to root (None)."""
    mocker.patch(
        "backend.api.features.workspace.folder_routes.get_or_create_workspace",
        AsyncMock(return_value=_make_workspace()),
    )
    move = mocker.patch(
        "backend.api.features.workspace.folder_routes.bulk_move_files_to_folder",
        AsyncMock(return_value=[]),
    )

    response = client.post("/folders/files/bulk-move", json={"file_ids": ["f1"]})
    assert response.status_code == 200
    move.assert_awaited_once_with(
        workspace_id="ws-001", file_ids=["f1"], folder_id=None
    )

from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from fastapi import HTTPException

from backend.api.features.workspace.view_state_routes import router
from backend.copilot.ui_view_files import UIFileStateResponse

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

_ROUTES = "backend.api.features.workspace.view_state_routes"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _state(**overrides) -> UIFileStateResponse:
    values = {
        "file_id": "file-1",
        "view_id": "trip",
        "source": 'root = Workspace("Trip", "", [], "trip")',
        "checksum": "abc123",
        "state": {"trip": {"notes": "Bridge"}},
        "revision": 2,
        "origin_session_id": None,
    }
    return UIFileStateResponse(**{**values, **overrides})


def test_get_returns_the_view_and_its_edits_for_the_signed_in_user(
    mocker, test_user_id
):
    read = mocker.patch(f"{_ROUTES}.read_file_state", AsyncMock(return_value=_state()))
    response = client.get("/files/file-1/view-state")
    assert response.status_code == 200
    assert response.json() == {
        "file_id": "file-1",
        "view_id": "trip",
        "source": 'root = Workspace("Trip", "", [], "trip")',
        "checksum": "abc123",
        "state": {"trip": {"notes": "Bridge"}},
        "revision": 2,
        "origin_session_id": None,
    }
    read.assert_awaited_once_with(test_user_id, "file-1")


def test_put_passes_revision_and_checksum(mocker, test_user_id):
    write = mocker.patch(
        f"{_ROUTES}.write_file_state", AsyncMock(return_value=_state(revision=3))
    )
    response = client.put(
        "/files/file-1/view-state",
        json={"expected_revision": 2, "expected_checksum": "abc123", "state": {}},
    )
    assert response.status_code == 200
    assert response.json()["revision"] == 3
    user_id, file_id, request = write.await_args.args
    assert (user_id, file_id) == (test_user_id, "file-1")
    assert (request.expected_revision, request.expected_checksum) == (2, "abc123")


@pytest.mark.parametrize(
    "body",
    [
        {"expected_revision": 0, "state": {}},
        {"expected_checksum": "abc123", "state": {}},
        {"expected_revision": 0, "expected_checksum": "", "state": {}},
        {
            "expected_revision": 0,
            "expected_checksum": "a",
            "state": {"x": "é" * 130_000},
        },
    ],
    ids=["no-checksum", "no-revision", "empty-checksum", "oversized"],
)
def test_put_requires_a_revision_a_checksum_and_bounded_state(mocker, body):
    write = mocker.patch(f"{_ROUTES}.write_file_state", AsyncMock())
    response = client.put("/files/file-1/view-state", json=body)
    assert response.status_code == 422
    write.assert_not_awaited()


@pytest.mark.parametrize("status", [404, 409])
def test_conflicts_and_missing_views_reach_the_client(mocker, status):
    mocker.patch(
        f"{_ROUTES}.write_file_state",
        AsyncMock(side_effect=HTTPException(status, "nope")),
    )
    response = client.put(
        "/files/file-1/view-state",
        json={"expected_revision": 0, "expected_checksum": "abc", "state": {}},
    )
    assert response.status_code == status

from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.chat.ui_drafts import router
from backend.copilot.ui_view_files import SavedUIViewResponse

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

_ROUTES = "backend.api.features.chat.ui_drafts"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_save_writes_the_view_to_a_file_named_by_the_user(mocker, test_user_id):
    saved = SavedUIViewResponse(
        file_id="file-1",
        file_name="Lisbon trip.openui",
        file_path="/sessions/chat-1/Lisbon trip.openui",
        file_uri="workspace://file-1#application/vnd.autogpt.openui",
        view_id="trip",
        checksum="abc123",
    )
    save = mocker.patch(f"{_ROUTES}.save_draft_as_file", AsyncMock(return_value=saved))
    response = client.post(
        "/sessions/chat-1/ui-drafts/trip/save", json={"filename": " Lisbon trip "}
    )
    assert response.status_code == 200
    assert response.json() == saved.model_dump()
    user_id, session_id, view_id, request = save.await_args.args
    assert (user_id, session_id, view_id) == (test_user_id, "chat-1", "trip")
    assert request.filename == "Lisbon trip.openui"


@pytest.mark.parametrize(
    "path, body",
    [
        ("/sessions/chat-1/ui-drafts/trip/save", {"filename": "../notes"}),
        ("/sessions/chat-1/ui-drafts/trip/save", {}),
        ("/sessions/chat-1/ui-drafts/1-bad/save", {"filename": "notes"}),
    ],
    ids=["path-in-name", "no-name", "bad-view-id"],
)
def test_save_rejects_bad_input(mocker, path, body):
    save = mocker.patch(f"{_ROUTES}.save_draft_as_file", AsyncMock())
    assert client.post(path, json=body).status_code == 422
    save.assert_not_awaited()

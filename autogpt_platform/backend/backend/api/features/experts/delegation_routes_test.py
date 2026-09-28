"""Delegation settings and the delegation list, under /experts."""

from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.features.experts import delegation_routes
from backend.api.features.experts.routes import router
from backend.copilot.delegation_settings import DelegationSettings

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

_SAVED = {
    "mode": "ask_first",
    "per_delegation_cap_usd": 5.0,
    "daily_budget_usd": 25.0,
    "ask_before_external": False,
    "ask_before_over_cap": True,
    "new_experts_ask_first": True,
}


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_settings_default_for_a_user_who_never_saved(
    mocker: pytest_mock.MockerFixture, test_user_id: str
):
    read = mocker.patch.object(
        delegation_routes.delegation_db,
        "get_delegation_settings",
        AsyncMock(return_value=DelegationSettings()),
    )

    response = client.get("/experts/delegation-settings")

    assert response.status_code == 200
    assert response.json() == DelegationSettings().model_dump()
    read.assert_awaited_once_with(test_user_id)


def test_saving_settings_stores_them_for_the_caller(
    mocker: pytest_mock.MockerFixture, test_user_id: str
):
    write = mocker.patch.object(
        delegation_routes.delegation_db,
        "update_delegation_settings",
        AsyncMock(side_effect=lambda user_id, settings: settings),
    )

    response = client.put("/experts/delegation-settings", json=_SAVED)

    assert response.status_code == 200
    assert response.json() == _SAVED
    assert write.await_args.args == (test_user_id, DelegationSettings(**_SAVED))


@pytest.mark.parametrize(
    "bad",
    [
        {"per_delegation_cap_usd": -1},
        {"daily_budget_usd": 1_000_000},
        {"mode": "yolo"},
    ],
)
def test_out_of_range_settings_are_refused(mocker: pytest_mock.MockerFixture, bad):
    write = mocker.patch.object(
        delegation_routes.delegation_db, "update_delegation_settings", AsyncMock()
    )

    response = client.put("/experts/delegation-settings", json={**_SAVED, **bad})

    assert response.status_code == 422
    write.assert_not_awaited()


def test_the_settings_route_is_not_taken_for_an_expert_id():
    """``/experts/{expert_id}`` must not swallow the settings path."""
    paths = [r.path for r in router.routes if isinstance(r, fastapi.routing.APIRoute)]

    assert paths.index("/experts/delegation-settings") < paths.index(
        "/experts/{expert_id}"
    )

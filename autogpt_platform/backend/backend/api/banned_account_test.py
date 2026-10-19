"""A banned account is refused on every route that resolves a user from a JWT."""

import fastapi
import pytest
from autogpt_libs.auth import (
    get_optional_user_id,
    get_request_context,
    get_user_id,
    requires_admin_user,
    requires_user,
)
from autogpt_libs.auth.dependencies import (
    ACCOUNT_BANNED_DETAIL,
    get_unbanned_jwt_payload,
)
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from backend.api.rest_api import app as rest_app


@pytest.mark.parametrize(
    "state, status",
    [("banned", 403), ("until_tomorrow", 403), ("lapsed", 200), ("unbanned", 200)],
)
def test_a_route_answers_by_the_tokens_ban_state(auth_row, state, status):
    user_id = auth_row(state)

    response = _client({"sub": user_id, "role": "user"}).get("/user-id")

    assert response.status_code == status
    if status == 403:
        assert response.json() == {"detail": ACCOUNT_BANNED_DETAIL}
    else:
        assert response.json() == {"user_id": user_id}


@pytest.mark.parametrize("path", ["/user-id", "/user", "/admin", "/context"])
def test_every_user_resolver_refuses_a_banned_account(auth_row, path):
    client = _client({"sub": auth_row("banned"), "role": "admin"})

    response = client.get(path)

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "account_banned"


def test_an_admin_can_still_act_as_a_banned_user(auth_row):
    banned = auth_row("banned")
    client = _client({"sub": auth_row("unbanned"), "role": "admin"})

    response = client.get("/user-id", headers={"X-Act-As-User-Id": banned})

    assert response.status_code == 200
    assert response.json() == {"user_id": banned}


@pytest.mark.parametrize("state", ["banned", "unbanned"])
def test_an_optional_auth_route_sees_a_banned_account_as_signed_out(
    mocker, auth_row, state
):
    user_id = auth_row(state)
    mocker.patch(
        "autogpt_libs.auth.jwt_utils.parse_jwt_token", return_value={"sub": user_id}
    )

    response = _client({}).get("/optional", headers={"Authorization": "Bearer t"})

    assert response.json() == {"user_id": None if state == "banned" else user_id}


def test_an_optional_auth_route_answers_when_the_ban_is_unreadable(mocker):
    mocker.patch("backend.data.db.is_connected", return_value=True)
    mocker.patch(
        "backend.data.user.is_user_banned",
        new=mocker.AsyncMock(side_effect=ConnectionError()),
    )
    mocker.patch(
        "autogpt_libs.auth.jwt_utils.parse_jwt_token", return_value={"sub": "u1"}
    )

    response = _client({}).get("/optional", headers={"Authorization": "Bearer t"})

    assert response.status_code == 200
    assert response.json() == {"user_id": None}


def test_every_route_that_reads_a_token_refuses_a_banned_account():
    """A route that takes ``get_jwt_payload`` without one of the resolvers
    above would let a banned account through."""
    unguarded = [
        f"{sorted(route.methods)} {route.path}"
        for route in rest_app.routes
        if isinstance(route, APIRoute)
        and get_jwt_payload in (calls := _dependency_calls(route.dependant))
        and get_unbanned_jwt_payload not in calls
    ]

    assert unguarded == []


def _client(jwt_payload: dict) -> TestClient:
    app = fastapi.FastAPI()

    @app.get("/user-id")
    async def by_user_id(user_id: str = fastapi.Security(get_user_id)):
        return {"user_id": user_id}

    @app.get("/user", dependencies=[fastapi.Security(requires_user)])
    async def by_user():
        return {}

    @app.get("/admin", dependencies=[fastapi.Security(requires_admin_user)])
    async def by_admin():
        return {}

    @app.get("/context", dependencies=[fastapi.Security(get_request_context)])
    async def by_context():
        return {}

    @app.get("/optional")
    async def by_optional(user_id=fastapi.Security(get_optional_user_id)):
        return {"user_id": user_id}

    app.dependency_overrides[get_jwt_payload] = lambda: jwt_payload
    return TestClient(app)


def _dependency_calls(dependant: Dependant) -> set:
    calls, stack = set(), list(dependant.dependencies)
    while stack:
        sub = stack.pop()
        calls.add(sub.call)
        stack.extend(sub.dependencies)
    return calls

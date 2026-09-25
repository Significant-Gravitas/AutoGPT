"""The swap proxy's service answers the two swap methods and nothing else."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route

from backend.copilot.swap_credentials import NoLiveBox, SwapCredential
from backend.copilot.swap_service import SwapCredentialService
from backend.data.db_manager import DatabaseManager
from backend.util.service import EXPOSED_FLAG

_SWAP_METHODS = {"get_swap_bindings", "resolve_swap_credential"}


@pytest.fixture(scope="module")
def client():
    # Not entered as a context manager: the lifespan (database, Redis) is not
    # what is under test, the routes are.
    return TestClient(SwapCredentialService().build_fastapi_app())


def _database_manager_methods() -> list[str]:
    return sorted(
        name
        for name, attr in vars(DatabaseManager).items()
        if getattr(attr, EXPOSED_FLAG, False)
    )


def test_the_only_rpc_routes_are_the_two_swap_methods():
    app = SwapCredentialService().build_fastapi_app()
    posts = {
        route.path.lstrip("/")
        for route in app.routes
        if isinstance(route, Route) and "POST" in (route.methods or ())
    }
    assert posts == _SWAP_METHODS | {"health_check", "health_check_async"}


def test_the_swap_methods_are_no_longer_on_database_manager():
    assert _SWAP_METHODS.isdisjoint(_database_manager_methods())


def test_every_database_manager_method_is_refused(client):
    methods = _database_manager_methods()
    assert "get_user_credentials" in methods  # the list is the real one
    answered = [
        method
        for method in methods
        if client.post(f"/{method}", json={"user_id": "user-1"}).status_code
        not in (404, 405)
    ]
    assert answered == []


def test_resolve_swap_credential_answers(client):
    credential = SwapCredential(
        name="github", values={"access_token": "ghp_x"}, allowed_hosts=["github.com"]
    )
    with patch(
        "backend.copilot.swap_service.resolve_swap_credential",
        AsyncMock(return_value=credential),
    ) as resolve:
        response = client.post(
            "/resolve_swap_credential",
            json={
                "user_id": "user-1",
                "name": "github",
                "host": "github.com",
                "box": "box-1",
            },
        )
    assert response.status_code == 200
    assert response.json() == credential.model_dump()
    resolve.assert_awaited_once_with("user-1", "github", "github.com", "box-1")


def test_resolve_swap_credential_answers_none(client):
    with patch(
        "backend.copilot.swap_service.resolve_swap_credential",
        AsyncMock(return_value=None),
    ):
        response = client.post(
            "/resolve_swap_credential",
            json={
                "user_id": "user-1",
                "name": "github",
                "host": "evil.test",
                "box": "box-1",
            },
        )
    assert response.status_code == 200
    assert response.json() is None


def test_get_swap_bindings_answers(client):
    response = client.post("/get_swap_bindings", json={})
    assert response.status_code == 200
    assert "github" in response.json()


def test_a_box_that_is_not_live_is_an_error_not_an_answer(client):
    """The proxy reads an error as "cannot say" and refuses that connection's
    responses; ``None`` would read as "not connected" and pass them on."""
    with patch(
        "backend.copilot.swap_service.resolve_swap_credential",
        AsyncMock(side_effect=NoLiveBox("no live box")),
    ):
        response = client.post(
            "/resolve_swap_credential",
            json={
                "user_id": "user-1",
                "name": "github",
                "host": "github.com",
                "box": "box-stale",
            },
        )
    assert response.status_code == 404

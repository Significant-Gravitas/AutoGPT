"""A request's PostHog events say whether an admin made it as the user, and
join the browser session it names; nothing carries over to the next request."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.middleware.analytics import AnalyticsContextMiddleware
from backend.util import posthog_client

SESSION = "0192f0c4-5f4e-7b4c-9d6e-1a2b3c4d5e6f"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.add_middleware(AnalyticsContextMiddleware)

    @app.get("/context")
    async def context():
        return {
            "impersonated": posthog_client.is_impersonated(),
            "session_id": posthog_client._session_id.get(),
        }

    @app.get("/sync-context")
    def sync_context():
        return {"impersonated": posthog_client.is_impersonated()}

    return TestClient(app)


def test_a_request_an_admin_makes_as_the_user_is_impersonated(client: TestClient):
    response = client.get("/context", headers={"X-Act-As-User-Id": "user-1"})
    assert response.json()["impersonated"] is True

    # A sync route runs on a worker thread, which sees the mark too.
    response = client.get("/sync-context", headers={"X-Act-As-User-Id": "user-1"})
    assert response.json()["impersonated"] is True


def test_a_users_own_request_is_not(client: TestClient):
    assert client.get("/context").json() == {
        "impersonated": False,
        "session_id": None,
    }
    assert (
        client.get("/context", headers={"X-Act-As-User-Id": " "}).json()["impersonated"]
        is False
    )


@pytest.mark.parametrize("value", [SESSION, SESSION.upper()])
def test_a_request_joins_the_browser_session_it_names(client: TestClient, value):
    response = client.get("/context", headers={"X-PostHog-Session-Id": value})
    assert response.json()["session_id"] == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not a session",
        "a" * 65,
        "<script>",
        # A UUIDv4, and a v7 with a bad variant.
        "0192f0c4-5f4e-4b4c-9d6e-1a2b3c4d5e6f",
        "0192f0c4-5f4e-7b4c-cd6e-1a2b3c4d5e6f",
    ],
)
def test_a_malformed_session_id_is_dropped(client: TestClient, value: str):
    response = client.get("/context", headers={"X-PostHog-Session-Id": value})
    assert response.json()["session_id"] is None


def test_nothing_carries_over_after_the_request(client: TestClient):
    client.get(
        "/context",
        headers={"X-Act-As-User-Id": "user-1", "X-PostHog-Session-Id": SESSION},
    )
    assert posthog_client.is_impersonated() is False
    assert posthog_client._session_id.get() is None

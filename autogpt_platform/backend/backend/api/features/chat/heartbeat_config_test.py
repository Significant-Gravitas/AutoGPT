"""Route tests for the heartbeat settings and the manual run; the heartbeat
itself is mocked at the module boundary (``copilot/heartbeat/*_test.py``
cover it)."""

from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.api.features.chat import heartbeat_config as heartbeat_routes
from backend.copilot.heartbeat.config import HeartbeatConfig
from backend.copilot.heartbeat.runner import HeartbeatRunResult

USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"

app = fastapi.FastAPI()
app.include_router(heartbeat_routes.router, prefix="/api/chat")
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def signed_in():
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    def payload(request: fastapi.Request) -> dict[str, str]:
        return {"sub": USER_ID, "role": "user", "email": "test@example.com"}

    app.dependency_overrides[get_jwt_payload] = payload
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def stored(mocker: pytest_mock.MockerFixture):
    saved: dict[str, HeartbeatConfig] = {}

    async def load(user_id: str) -> HeartbeatConfig:
        return saved.get(user_id, HeartbeatConfig())

    async def save(user_id: str, config: HeartbeatConfig) -> None:
        saved[user_id] = config

    mocker.patch.object(heartbeat_routes, "load_config", side_effect=load)
    mocker.patch.object(heartbeat_routes, "save_config", side_effect=save)
    mocker.patch.object(
        heartbeat_routes, "resolve_timezone", AsyncMock(return_value="Europe/Berlin")
    )
    return saved


def test_unsaved_settings_read_as_the_defaults(stored):
    response = client.get("/api/chat/heartbeat")
    assert response.status_code == 200
    body = response.json()
    assert body["config"]["enabled"] is False
    assert body["config"]["interval_minutes"] == 30
    assert body["effective_timezone"] == "Europe/Berlin"
    assert body["scheduled"] is False
    assert body["checklist_empty"] is True


def test_saving_registers_the_job_and_resets_the_change_signal(
    stored, mocker: pytest_mock.MockerFixture
):
    sync = mocker.patch.object(
        heartbeat_routes, "sync_heartbeat_schedule", AsyncMock(return_value=True)
    )
    clear = mocker.patch.object(heartbeat_routes.state, "clear_last_run", AsyncMock())
    response = client.put(
        "/api/chat/heartbeat",
        json={
            "enabled": True,
            "interval_minutes": 45,
            "checklist": "- Tell me if a run failed",
            "delivery": {"chat_platforms": ["discord"]},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scheduled"] is True and body["schedule_synced"] is True
    assert stored[USER_ID].interval_minutes == 45
    assert stored[USER_ID].delivery.chat_platforms == ["discord"]
    sync.assert_awaited_once_with(USER_ID, stored[USER_ID])
    clear.assert_awaited_once_with(USER_ID)


def test_invalid_settings_are_refused(stored):
    response = client.put(
        "/api/chat/heartbeat", json={"enabled": True, "active_hours_start": "25:00"}
    )
    assert response.status_code == 422
    assert USER_ID not in stored


def test_run_now_forces_a_beat(mocker: pytest_mock.MockerFixture):
    mocker.patch.object(
        heartbeat_routes.state, "claim_manual_run", AsyncMock(return_value=True)
    )
    run = mocker.patch.object(
        heartbeat_routes,
        "run_heartbeat",
        AsyncMock(
            return_value=HeartbeatRunResult(
                status="silent", reason="silent", session_id="hb"
            )
        ),
    )
    response = client.post("/api/chat/heartbeat/run")
    assert response.status_code == 200
    assert response.json()["status"] == "silent"
    run.assert_awaited_once_with(USER_ID, force=True)


def test_run_now_is_rate_limited(mocker: pytest_mock.MockerFixture):
    mocker.patch.object(
        heartbeat_routes.state, "claim_manual_run", AsyncMock(return_value=False)
    )
    run = mocker.patch.object(heartbeat_routes, "run_heartbeat", AsyncMock())
    assert client.post("/api/chat/heartbeat/run").status_code == 429
    run.assert_not_awaited()


def test_routes_need_a_user():
    app.dependency_overrides.clear()
    assert client.get("/api/chat/heartbeat").status_code == 401
